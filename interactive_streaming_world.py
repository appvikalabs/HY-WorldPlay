#!/usr/bin/env python3
"""
vMonad - Interactive World Streaming Demo

Why this exists
---------------
The paper's continuity WITHOUT degradation comes from keeping the *latent* world state and
the transformer's KV-cache + Reconstituted Context Memory across chunk rollouts.

If you decode frames to RGB and re-encode them as the next reference image, you will drift/blur.
If you regenerate from the original reference each time, you'll replay the same relative clip.

This demo:
- Loads the pipeline once
- Encodes prompt/vision/byt5 once
- Maintains:
    - generated latent video state (latent frames)
    - camera poses (w2c) + intrinsics (K) + discrete action labels
    - transformer KV cache (text cached once, vision cached per chunk from retrieved memory)
- Generates ONE latent-chunk at a time (default: 4 latent frames)
- Decodes and displays only the NEW video frames for smooth continuation

Controls
--------
- Hold WASD / Arrow keys to choose motion
- SPACE: generate next chunk
- R: reset stream state (keeps models loaded)
- ESC: quit

Notes
-----
- First chunk displays 13 frames (4 latent frames -> (4-1)*4+1 = 13)
- Each subsequent chunk adds 16 frames (because latent length +4 => video length +16)
"""

import os
import time
import threading
import queue
import traceback
import json
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Make distributed/device-mesh init deterministic for single-GPU runs.
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import numpy as np
import torch
import pygame
from PIL import Image
from scipy.spatial.transform import Rotation as R

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.pipelines.pipeline_utils import retrieve_timesteps, rescale_noise_cfg
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons import is_flash_available
from hyvideo.utils.retrieval_context import generate_points_in_sphere, select_aligned_memory_frames


# -----------------------------
# Config (tweakable)
# -----------------------------

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 368  # must be divisible by 16

CHUNK_LATENT_FRAMES = 4  # 4 latents => 13 frames initially; +4 latents => +16 frames
NUM_INFERENCE_STEPS = 4  # distilled
FLOW_SHIFT = 5.0
SEED = 42

# Camera control (bigger = faster exploration, but more blur in newly revealed regions)
MOVE_STEP_PRESETS = [0.03, 0.05, 0.08]
YAW_STEP_DEG = 3.0
PITCH_STEP_DEG = 2.0
DEFAULT_MOVE_STEP_IDX = 1

# Memory retrieval config (matches pipeline defaults)
MEMORY_FRAMES = 20
TEMPORAL_CONTEXT_SIZE = 12
PRED_LATENT_SIZE = 4
STABILIZATION_LEVEL = 15

# Monte-Carlo points for FOV overlap similarity. Bigger = better retrieval, slower.
POINTS_LOCAL_N = 8192
POINTS_LOCAL_RADIUS = 8.0

# Display config
DISPLAY_SCALE = 2
FPS_PLAYBACK = 24

# UI scaling: `scale` is fastest but can introduce moiré/"web" patterns when upscaling.
# `smoothscale` is higher quality (slightly more CPU).
USE_SMOOTHSCALE = True

# Session recording (pose JSON + meta for batch re-render)
REPETITIONS_DIR = "repetitions"

# Debug prints can be noisy; set True if needed.
DEBUG = False


# -----------------------------
# Runtime prompt/effect injection
# -----------------------------
#
# The paper’s “prompt injection” for streaming systems is best approximated here by:
# - re-encoding the prompt (text + byt5)
# - refreshing the cached text KV
# while KEEPING the latent world state + memory lists.
#
# IMPORTANT: In 4-step distilled mode, prompt-following can be weak. Two practical levers:
# - Provide very explicit, visual prompts for effects.
# - Optionally enable CFG (slower but much stronger prompt adherence).
# NOTE: CFG too high can produce visible artifacts in streaming. Keep this conservative.
GUIDANCE_SCALES: List[float] = [1.0, 2.0]  # 1.0 = fastest, 2.0 = stronger prompt adherence (CFG)
GUIDANCE_RESCALE: float = 0.0

# If the user asks for common global effects (rain/fog/smoke) and CFG is off,
# we can auto-boost CFG to make the change visible.
AUTO_BOOST_CFG_ON_EFFECT: bool = False

_EFFECT_PROMPTS: Dict[str, str] = {
    # Global, scene-wide effects work best for prompt injection.
    "rain": (
        "It starts raining heavily. Visible rain streaks and raindrops across the entire frame. "
        "Wet surfaces, puddles, reflections, overcast sky. "
        "开始下大雨，雨丝雨滴清晰可见，地面潮湿有积水和反光，空气中有雨雾。"
    ),
    "fog": (
        "Dense fog/mist appears. Volumetric fog, atmospheric haze, reduced visibility, light scattering. "
        "出现浓雾/薄雾，空气中有体积雾与雾气，能见度降低，光线散射。"
    ),
    "smoke": (
        "Thick smoke drifts through the scene. Volumetric smoke, floating particles, atmospheric haze. "
        "烟雾在画面中飘动，体积烟雾与漂浮颗粒，空气中有雾霾感。"
    ),
}

_EFFECT_NEGATIVE_PROMPTS: Dict[str, str] = {
    "rain": "sunny, clear blue sky, dry ground, no rain, no puddles",
    "fog": "crystal clear air, perfectly clear visibility, no fog, no haze",
    "smoke": "clean air, no smoke, no haze, no particles",
}


def _detect_effect_keywords(text: str) -> List[str]:
    t = (text or "").lower()
    effects = set()
    if any(k in t for k in ["rain", "raining", "rainy"]) or any(k in (text or "") for k in ["雨", "下雨", "大雨"]):
        effects.add("rain")
    if any(k in t for k in ["fog", "mist", "haze", "smog"]) or any(k in (text or "") for k in ["雾", "雾气", "大雾", "薄雾"]):
        effects.add("fog")
    if any(k in t for k in ["smoke", "smoky", "smoking"]) or any(k in (text or "") for k in ["烟", "烟雾", "烟气"]):
        effects.add("smoke")
    return sorted(effects)


def _expand_injection_text(user_text: str) -> Tuple[str, List[str]]:
    """
    Expand short/ambiguous effect prompts into explicit visual descriptors to increase adherence.
    Returns: (expanded_text, detected_effects)
    """
    user_text = (user_text or "").strip()
    detected = _detect_effect_keywords(user_text)
    parts: List[str] = []
    if user_text:
        parts.append(user_text)
    for eff in detected:
        parts.append(_EFFECT_PROMPTS.get(eff, ""))
    expanded = " ".join([p for p in parts if p]).strip()
    return expanded, detected


def build_injected_prompt(base_prompt: str, enabled_effects: List[str], custom_injection: Optional[str]) -> Tuple[str, List[str]]:
    """
    Build the effective prompt for the next chunks.
    Returns: (prompt, all_effects_detected)
    """
    enabled_effects = [e for e in (enabled_effects or []) if e in _EFFECT_PROMPTS]
    custom_injection = (custom_injection or "").strip()

    expanded_custom = ""
    detected_from_custom: List[str] = []
    if custom_injection:
        expanded_custom, detected_from_custom = _expand_injection_text(custom_injection)

    all_effects = sorted(set(enabled_effects) | set(detected_from_custom))

    # If nothing is injected, return base prompt verbatim.
    if not all_effects and not expanded_custom:
        return base_prompt, all_effects

    parts: List[str] = [base_prompt, "Keep the same scene layout and camera trajectory."]
    if all_effects:
        parts.append(" ".join(_EFFECT_PROMPTS[e] for e in all_effects if e in _EFFECT_PROMPTS))
    if expanded_custom:
        parts.append(expanded_custom)
    return " ".join(parts).strip(), all_effects


def build_negative_prompt(effects: List[str]) -> str:
    effects = [e for e in (effects or []) if e in _EFFECT_NEGATIVE_PROMPTS]
    if not effects:
        return ""
    return " ".join(_EFFECT_NEGATIVE_PROMPTS[e] for e in effects if e in _EFFECT_NEGATIVE_PROMPTS).strip()


def apply_prompt_injection(state: "StreamState", new_prompt: str, negative_prompt: str = "") -> None:
    """
    Update prompt embeddings + byt5 embeddings and refresh cached text KV.
    Does NOT reset latent world state.
    """
    pipe = state.pipe
    device = state.device

    with torch.no_grad():
        (
            prompt_embeds,
            _neg_prompt_embeds,
            prompt_mask,
            _neg_prompt_mask,
        ) = pipe.encode_prompt(
            new_prompt,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            negative_prompt=negative_prompt if pipe.do_classifier_free_guidance else "",
            clip_skip=None,
            data_type="video",
        )

        extra_kwargs: Dict[str, torch.Tensor] = {}
        if pipe.config.glyph_byT5_v2:
            extra_kwargs = pipe._prepare_byt5_embeddings(new_prompt, device)

    # Match the pipeline behavior: concatenate negative+positive when CFG is enabled.
    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([_neg_prompt_embeds, prompt_embeds], dim=0)
        if prompt_mask is not None and _neg_prompt_mask is not None:
            prompt_mask = torch.cat([_neg_prompt_mask, prompt_mask], dim=0)

    state.prompt_text = new_prompt
    state.negative_prompt_text = negative_prompt
    state.prompt_embeds = prompt_embeds
    state.prompt_mask = prompt_mask
    state.extra_kwargs = extra_kwargs

    # Ensure vision states batch dimension matches CFG mode.
    if state.vision_states_single is not None:
        state.vision_states = (
            state.vision_states_single.repeat(2, 1, 1) if pipe.do_classifier_free_guidance else state.vision_states_single
        )

    _cache_text_once(state)
    # record prompt changes (if enabled)
    if state.recorder is not None:
        state.recorder.log_event("prompt_injection", prompt=new_prompt, negative_prompt=negative_prompt)


# -----------------------------
# Action encoding (same mapping as generate.py)
# -----------------------------

_ONEHOT4_TO_LABEL = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def action_label_from_keys(move: str, turn: Optional[str]) -> int:
    # translation one-hot: [forward, backward, right, left]
    trans = [0, 0, 0, 0]
    if move == "forward":
        trans[0] = 1
    elif move == "backward":
        trans[1] = 1
    elif move == "right":
        trans[2] = 1
    elif move == "left":
        trans[3] = 1

    # rotation one-hot: [right, left, up, down]
    rot = [0, 0, 0, 0]
    if turn == "right":
        rot[0] = 1
    elif turn == "left":
        rot[1] = 1
    elif turn == "up":
        rot[2] = 1
    elif turn == "down":
        rot[3] = 1

    trans_label = _ONEHOT4_TO_LABEL[tuple(trans)]
    rot_label = _ONEHOT4_TO_LABEL[tuple(rot)]
    return trans_label * 9 + rot_label


# -----------------------------
# Camera trajectory helper
# -----------------------------

@dataclass
class CameraState:
    c2w: np.ndarray  # 4x4


def normalized_K(width: int, height: int) -> np.ndarray:
    """
    Pipeline expects Ks shaped (B, T, 3, 3) with cx=cy=0.5 and fx,fy normalized by width/height.
    This matches generate.py's pose_to_input() output after normalization.
    """
    fx = fy = 969.6969697
    fx = fx / float(width)
    fy = fy / float(height)
    return np.array([[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def unnormalized_K(width: int, height: int) -> np.ndarray:
    """
    Batch pose JSON (consumed by generate.py) expects *unnormalized* intrinsics with cx=width/2, cy=height/2.
    generate.py will normalize them (fx/width, fy/height) and set cx=cy=0.5 internally.
    """
    fx = fy = 969.6969697
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass
class SegmentRecorder:
    """
    Records ONE interactive segment (one reference-image set + continuous camera trajectory).
    Writes:
      - repetitions/<segment_id>.json      (batch-compatible pose JSON for generate.py)
      - repetitions/<segment_id>_meta.json (prompts, settings, action labels, events)
      - repetitions/<segment_id>_img*.ext  (copies of reference images)
    """

    segment_id: str
    created_at_iso: str
    width: int
    height: int
    model_path: str
    action_ckpt: str
    reference_image_paths: List[str]
    base_prompt: str
    negative_prompt: str
    flow_shift: float
    seed: int
    notes: str = ""

    # trajectory: one entry per latent step
    c2w_list: List[np.ndarray] = None
    action_labels: List[int] = None
    events: List[Dict[str, Any]] = None

    def __post_init__(self):
        self.c2w_list = [] if self.c2w_list is None else self.c2w_list
        self.action_labels = [] if self.action_labels is None else self.action_labels
        self.events = [] if self.events is None else self.events

    @staticmethod
    def new(
        *,
        width: int,
        height: int,
        model_path: str,
        action_ckpt: str,
        reference_image_paths: List[str],
        base_prompt: str,
        negative_prompt: str,
        flow_shift: float,
        seed: int,
        notes: str = "",
    ) -> "SegmentRecorder":
        sid = datetime.now().strftime("%Y%m%d-%H%M%S")
        return SegmentRecorder(
            segment_id=sid,
            created_at_iso=datetime.now().isoformat(timespec="seconds"),
            width=width,
            height=height,
            model_path=model_path,
            action_ckpt=action_ckpt,
            reference_image_paths=reference_image_paths,
            base_prompt=base_prompt,
            negative_prompt=negative_prompt,
            flow_shift=flow_shift,
            seed=seed,
            notes=notes,
        )

    def log_event(self, typ: str, **data: Any) -> None:
        self.events.append(
            {"t": datetime.now().isoformat(timespec="seconds"), "type": typ, "data": data}
        )

    def append_step(self, c2w: np.ndarray, action_label: int) -> None:
        self.c2w_list.append(np.array(c2w, dtype=np.float32))
        self.action_labels.append(int(action_label))

    def save(self) -> Tuple[str, str]:
        out_dir = Path(REPETITIONS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        copied_images: List[str] = []
        for i, src in enumerate(self.reference_image_paths):
            try:
                src_path = Path(src)
                ext = src_path.suffix if src_path.suffix else ".png"
                dst = out_dir / f"{self.segment_id}_img{i}{ext}"
                shutil.copyfile(src_path, dst)
                copied_images.append(str(dst))
            except Exception as e:
                self.log_event("warn_image_copy_failed", index=i, src=src, error=str(e))

        # Batch-compatible pose JSON for generate.py (ONLY numeric keys at top-level)
        K_raw = unnormalized_K(self.width, self.height).tolist()
        pose_json: Dict[str, Dict[str, Any]] = {}
        for i, c2w in enumerate(self.c2w_list):
            pose_json[str(i)] = {"extrinsic": c2w.tolist(), "K": K_raw}

        pose_path = out_dir / f"{self.segment_id}.json"
        with open(pose_path, "w") as f:
            json.dump(pose_json, f, indent=2)

        meta = {
            "segment_id": self.segment_id,
            "created_at": self.created_at_iso,
            "width": self.width,
            "height": self.height,
            "model_path": self.model_path,
            "action_ckpt": self.action_ckpt,
            "reference_image_paths": self.reference_image_paths,
            "copied_reference_images": copied_images,
            "base_prompt": self.base_prompt,
            "negative_prompt": self.negative_prompt,
            "flow_shift": self.flow_shift,
            "seed": self.seed,
            "num_latent_steps": len(self.c2w_list),
            "action_labels": self.action_labels,
            "events": self.events,
            "notes": self.notes,
            "batch_render_hint": {
                "pose_json_path": str(pose_path),
                "image_path": copied_images[0] if copied_images else (self.reference_image_paths[0] if self.reference_image_paths else None),
                "example_generate_py": (
                    "torchrun --nproc_per_node=1 generate.py "
                    f"--pose_json_path {pose_path} "
                    "--image_path <image_path> "
                    "--prompt \"<prompt>\" "
                    "--video_length <frames> "
                    "--num_inference_steps <steps> "
                    "--few_step <true|false>"
                ),
            },
        }

        meta_path = out_dir / f"{self.segment_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return str(pose_path), str(meta_path)

def step_camera_pose(c2w: np.ndarray, move: str, turn: Optional[str]) -> np.ndarray:
    """
    Apply one latent-step of camera movement in local frame, similar scale to test_forward pose.
    """
    # These are tuned for streaming stability; too-large steps jump into unseen regions
    # and the model has to hallucinate/inpaint, which looks blurry.
    move_step = MOVE_STEP_PRESETS[DEFAULT_MOVE_STEP_IDX]
    yaw_step_deg = YAW_STEP_DEG
    pitch_step_deg = PITCH_STEP_DEG

    out = c2w.copy()

    # Rotation (local)
    if turn is not None:
        if turn == "left":
            rot = R.from_euler("y", -yaw_step_deg, degrees=True).as_matrix()
        elif turn == "right":
            rot = R.from_euler("y", yaw_step_deg, degrees=True).as_matrix()
        elif turn == "up":
            rot = R.from_euler("x", -pitch_step_deg, degrees=True).as_matrix()
        elif turn == "down":
            rot = R.from_euler("x", pitch_step_deg, degrees=True).as_matrix()
        else:
            rot = np.eye(3)
        out[:3, :3] = out[:3, :3] @ rot

    # Translation (local -> world)
    local = np.zeros(3, dtype=np.float32)
    if move == "forward":
        local[2] = move_step
    elif move == "backward":
        local[2] = -move_step
    elif move == "right":
        local[0] = move_step
    elif move == "left":
        local[0] = -move_step

    world = out[:3, :3] @ local
    out[:3, 3] = out[:3, 3] + world

    return out


# -----------------------------
# Streaming state + rollout
# -----------------------------

@dataclass
class StreamState:
    # static
    pipe: HunyuanVideo_1_5_Pipeline
    device: torch.device
    generator: torch.Generator
    prompt_embeds: torch.Tensor
    prompt_mask: Optional[torch.Tensor]
    vision_states: Optional[torch.Tensor]
    # Always keep the single (positive) vision state to support toggling CFG without re-encoding.
    vision_states_single: Optional[torch.Tensor]
    extra_kwargs: Dict[str, torch.Tensor]
    timesteps: torch.Tensor
    points_local: torch.Tensor
    task_type: str = "i2v"
    prompt_text: str = ""
    negative_prompt_text: str = ""
    # Quality/speed knobs (captured once per generated chunk)
    num_inference_steps: int = NUM_INFERENCE_STEPS
    # Recording (optional)
    recorder: Optional[SegmentRecorder] = None

    # dynamic
    c2w_list: List[np.ndarray] = None
    w2c_list: List[np.ndarray] = None
    K_list: List[np.ndarray] = None
    action_list: List[int] = None

    latents: Optional[torch.Tensor] = None          # [1, C, T, h, w]
    cond_latents: Optional[torch.Tensor] = None     # [1, C+1, T, h, w] (cond + mask)


def init_stream(
    model_path: str,
    action_ckpt: str,
    prompt: str,
    image_path: str,
    width: int,
    height: int,
) -> Tuple[StreamState, np.ndarray]:
    import time
    t0 = time.perf_counter()
    # parallel + infer state (single GPU)
    initialize_parallel_state(sp=1)

    class InferArgs:
        def __init__(self):
            self.enable_torch_compile = False

    initialize_infer_state(InferArgs())
    torch.cuda.set_device(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t_pipe0 = time.perf_counter()
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=model_path,
        transformer_version="480p_i2v",
        enable_offloading=False,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=action_ckpt,
    )
    t_pipe1 = time.perf_counter()
    print(f"[startup] create_pipeline: {(t_pipe1 - t_pipe0):.2f}s")

    # Use SageAttention if available (faster 8-bit attention as per paper),
    # otherwise fall back to PyTorch SDPA.
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        try:
            from sageattention import sageattn  # noqa: F401
            pipe.transformer.set_attn_mode("sageattn")
            print("Using SageAttention (8-bit quantized attention)")
        except ImportError:
            if not is_flash_available():
                pipe.transformer.set_attn_mode("torch")
                print("Using PyTorch SDPA (flash-attn not available)")

    # keep models on GPU
    t_to0 = time.perf_counter()
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        pipe.transformer.to(device)
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to(device)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if hasattr(pipe, "byt5_model") and pipe.byt5_model is not None:
        pipe.byt5_model.to(device)
    if hasattr(pipe, "vision_encoder") and pipe.vision_encoder is not None:
        pipe.vision_encoder.to(device)
    t_to1 = time.perf_counter()
    print(f"[startup] move models to GPU: {(t_to1 - t_to0):.2f}s")

    # Default prompt strength (can be toggled at runtime with 'G' in the UI).
    pipe._guidance_scale = GUIDANCE_SCALES[0]
    pipe._guidance_rescale = GUIDANCE_RESCALE

    # generator
    gen = torch.Generator(device=device).manual_seed(SEED)

    # Prepare an initial dummy latents tensor for shape-dependent encoders
    latent_len = CHUNK_LATENT_FRAMES
    latent_len, latent_h, latent_w = pipe.get_latent_size(
        video_length=(latent_len - 1) * pipe.vae_temporal_compression_ratio + 1,
        height=height,
        width=width,
    )
    assert latent_len == CHUNK_LATENT_FRAMES
    latents_dummy = torch.zeros(
        (1, pipe.transformer.config.in_channels, latent_len, latent_h, latent_w),
        device=device,
        dtype=pipe.target_dtype,
    )

    # reference image
    t_img0 = time.perf_counter()
    reference_image = Image.open(image_path).convert("RGB")
    reference_np = np.array(reference_image)
    t_img1 = time.perf_counter()
    print(f"[startup] load reference image: {(t_img1 - t_img0):.2f}s")

    # text embeddings
    t_enc0 = time.perf_counter()
    with torch.no_grad():
        (
            prompt_embeds,
            _neg_prompt_embeds,
            prompt_mask,
            _neg_prompt_mask,
        ) = pipe.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            negative_prompt="",
            clip_skip=None,
            data_type="video",
        )

        extra_kwargs = {}
        if pipe.config.glyph_byT5_v2:
            extra_kwargs = pipe._prepare_byt5_embeddings(prompt, device)

        vision_states = pipe._prepare_vision_states(reference_np, pipe.ideal_resolution, latents_dummy, device)
        vision_states_single = vision_states
        if pipe.do_classifier_free_guidance and vision_states is not None:
            vision_states = vision_states.repeat(2, 1, 1)

        # scheduler timesteps (fixed per chunk)
        pipe.scheduler = pipe._create_scheduler(FLOW_SHIFT)
        timesteps, _ = retrieve_timesteps(pipe.scheduler, NUM_INFERENCE_STEPS, device=device)
    t_enc1 = time.perf_counter()
    print(f"[startup] encode prompt + vision + scheduler: {(t_enc1 - t_enc0):.2f}s")

    # points for memory retrieval
    points_local = generate_points_in_sphere(POINTS_LOCAL_N, POINTS_LOCAL_RADIUS).to(device)

    # init dynamic lists
    I = np.eye(4, dtype=np.float32)
    K = normalized_K(width, height)

    state = StreamState(
        pipe=pipe,
        device=device,
        generator=gen,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        vision_states=vision_states,
        vision_states_single=vision_states_single,
        extra_kwargs=extra_kwargs,
        timesteps=timesteps,
        points_local=points_local,
        prompt_text=prompt,
        negative_prompt_text="",
        c2w_list=[I],
        w2c_list=[np.linalg.inv(I)],
        K_list=[K],
        action_list=[0],
        latents=None,
        cond_latents=None,
    )

    # init KV cache + cache text once (same as ar_rollout first part)
    _cache_text_once(state)

    # initial display frame
    ref_display = reference_image.resize((width, height))
    frame0 = np.array(ref_display, dtype=np.uint8)

    return state, frame0


def _cache_text_once(state: StreamState) -> None:
    pipe = state.pipe
    pipe.init_kv_cache()

    # ar_rollout assumes byt5 embeddings exist; keep consistent
    extra_kwargs = state.extra_kwargs
    if "byt5_text_states" not in extra_kwargs or "byt5_text_mask" not in extra_kwargs:
        raise RuntimeError("ByT5 embeddings missing; pipeline expects glyph_byT5_v2=True for WorldPlay.")

    positive_idx = 1 if pipe.do_classifier_free_guidance else 0

    def _extra_kwargs_at(idx: int) -> Dict[str, torch.Tensor]:
        return {
            "byt5_text_states": extra_kwargs["byt5_text_states"][idx, None, ...],
            "byt5_text_mask": extra_kwargs["byt5_text_mask"][idx, None, ...],
        }

    t_expand_txt = torch.tensor([0], device=state.device, dtype=pipe.target_dtype)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipe.target_dtype, enabled=pipe.autocast_enabled):
        pipe._kv_cache = pipe.transformer(
            bi_inference=False,
            ar_txt_inference=True,
            ar_vision_inference=False,
            timestep_txt=t_expand_txt,
            text_states=state.prompt_embeds[positive_idx, None, ...],
            encoder_attention_mask=state.prompt_mask[positive_idx, None, ...] if state.prompt_mask is not None else None,
            vision_states=state.vision_states[positive_idx, None, ...] if state.vision_states is not None else None,
            mask_type=state.task_type,
            extra_kwargs=_extra_kwargs_at(positive_idx),
            kv_cache=pipe._kv_cache,
            cache_txt=True,
        )

        if pipe.do_classifier_free_guidance:
            pipe._kv_cache_neg = pipe.transformer(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=t_expand_txt,
                text_states=state.prompt_embeds[0, None, ...],
                encoder_attention_mask=state.prompt_mask[0, None, ...] if state.prompt_mask is not None else None,
                vision_states=state.vision_states[0, None, ...] if state.vision_states is not None else None,
                mask_type=state.task_type,
                extra_kwargs=_extra_kwargs_at(0),
                kv_cache=pipe._kv_cache_neg,
                cache_txt=True,
            )


def _append_cond_latents_for_new_frames(state: StreamState, image_cond: Optional[torch.Tensor]) -> None:
    """
    Maintain state.cond_latents the same way pipeline does:
    - for global frame 0: has reference latent + mask=1
    - for all other frames: cond=0, mask=0
    """
    pipe = state.pipe
    device = state.device
    C = pipe.transformer.config.in_channels

    T = state.latents.shape[2]
    assert T % CHUNK_LATENT_FRAMES == 0

    if state.cond_latents is None:
        # first chunk: create exactly like pipeline._prepare_cond_latents
        multitask_mask = pipe.get_task_mask("i2v", CHUNK_LATENT_FRAMES)
        cond_latents = pipe._prepare_cond_latents("i2v", image_cond, state.latents[:, :, :CHUNK_LATENT_FRAMES], multitask_mask)
        state.cond_latents = cond_latents.to(device=device, dtype=pipe.target_dtype)
    else:
        # subsequent chunks: append zeros (cond+mask)
        zeros = torch.zeros((1, C + 1, CHUNK_LATENT_FRAMES, state.latents.shape[3], state.latents.shape[4]),
                            device=device, dtype=pipe.target_dtype)
        state.cond_latents = torch.cat([state.cond_latents, zeros], dim=2)


def _append_pose_action_chunk(state: StreamState, move: str, turn: Optional[str]) -> None:
    """
    Append CHUNK_LATENT_FRAMES poses and actions.
    For the very first chunk, state already has pose[0] and action[0]=0.
    We append poses until total length matches latents length.
    """
    # we already have at least one pose in state lists
    while len(state.c2w_list) < state.latents.shape[2]:
        prev = state.c2w_list[-1]
        nxt = step_camera_pose(prev, move, turn)

        state.c2w_list.append(nxt)
        state.w2c_list.append(np.linalg.inv(nxt))
        state.K_list.append(state.K_list[0])  # constant intrinsics
        a = action_label_from_keys(move, turn)
        state.action_list.append(a)
        if state.recorder is not None:
            state.recorder.append_step(nxt, a)


def _stack_inputs(state: StreamState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    viewmats = torch.tensor(np.stack(state.w2c_list, axis=0), dtype=torch.float32, device=state.device).unsqueeze(0)
    Ks = torch.tensor(np.stack(state.K_list, axis=0), dtype=torch.float32, device=state.device).unsqueeze(0)
    action = torch.tensor(np.array(state.action_list), dtype=torch.float32, device=state.device).unsqueeze(0)
    return viewmats, Ks, action


def generate_next_chunk(state: StreamState, move: str, turn: Optional[str], reference_image_path: str, height: int, width: int) -> np.ndarray:
    """
    Generate one more chunk and return NEW RGB frames as uint8: [F, H, W, 3]
    """
    pipe = state.pipe
    device = state.device
    do_cfg = pipe.do_classifier_free_guidance
    steps = int(getattr(state, "num_inference_steps", NUM_INFERENCE_STEPS))

    # IMPORTANT: reset scheduler state for every chunk.
    # FlowMatchDiscreteScheduler increments an internal `step_index` on each `step()`.
    # Without resetting it, the 2nd call to `generate_next_chunk()` will start at the
    # end of the schedule and crash with an out-of-bounds sigma access.
    pipe.scheduler.set_timesteps(steps, device=device)
    state.timesteps = pipe.scheduler.timesteps

    # latent spatial sizes
    _, latent_h, latent_w = pipe.get_latent_size(video_length=13, height=height, width=width)

    # append noise latents
    first_chunk = state.latents is None
    new_latents = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        latent_h,
        latent_w,
        CHUNK_LATENT_FRAMES,
        pipe.target_dtype,
        device,
        state.generator,
    )
    if state.latents is None:
        state.latents = new_latents
    else:
        state.latents = torch.cat([state.latents, new_latents], dim=2)

    # ensure cond_latents
    if first_chunk:
        reference_image = Image.open(reference_image_path).convert("RGB")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            image_cond = pipe.get_image_condition_latents("i2v", reference_image, height, width).to(device)
        _append_cond_latents_for_new_frames(state, image_cond=image_cond)
    else:
        _append_cond_latents_for_new_frames(state, image_cond=None)

    # ensure poses/actions length matches latents length
    _append_pose_action_chunk(state, move=move, turn=turn)

    viewmats, Ks, action = _stack_inputs(state)

    # chunk index
    total_latents = state.latents.shape[2]
    chunk_i = (total_latents // CHUNK_LATENT_FRAMES) - 1
    start_idx = chunk_i * CHUNK_LATENT_FRAMES
    end_idx = start_idx + CHUNK_LATENT_FRAMES

    selected_frame_indices: List[int] = []

    # Reconstituted Context Memory (same as ar_rollout)
    if chunk_i > 0:
        current_frame_idx = start_idx
        for chunk_start_idx in range(current_frame_idx, current_frame_idx + CHUNK_LATENT_FRAMES, 4):
            selected_history_frame_id = select_aligned_memory_frames(
                viewmats[0].detach().cpu().numpy(),
                chunk_start_idx,
                memory_frames=MEMORY_FRAMES,
                temporal_context_size=TEMPORAL_CONTEXT_SIZE,
                pred_latent_size=PRED_LATENT_SIZE,
                points_local=state.points_local,
                device=device,
            )
            selected_frame_indices += selected_history_frame_id
        selected_frame_indices = sorted(list(set(selected_frame_indices)))
        to_remove = list(range(current_frame_idx, current_frame_idx + CHUNK_LATENT_FRAMES))
        selected_frame_indices = [x for x in selected_frame_indices if x not in to_remove]

        # cache vision kv for retrieved context
        context_latents = state.latents[:, :, selected_frame_indices]
        context_cond = state.cond_latents[:, :, selected_frame_indices]
        context_input = torch.concat([context_latents, context_cond], dim=1)

        context_viewmats = viewmats[:, selected_frame_indices]
        context_Ks = Ks[:, selected_frame_indices]
        context_action = action[:, selected_frame_indices]

        context_timestep = torch.full(
            (len(selected_frame_indices),),
            STABILIZATION_LEVEL - 1,
            device=device,
            dtype=state.timesteps.dtype,
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipe.target_dtype, enabled=pipe.autocast_enabled):
            pipe._kv_cache = pipe.transformer(
                bi_inference=False,
                ar_txt_inference=False,
                ar_vision_inference=True,
                hidden_states=context_input,
                timestep=context_timestep,
                timestep_r=None,
                mask_type=state.task_type,
                return_dict=False,
                viewmats=context_viewmats.to(pipe.target_dtype),
                Ks=context_Ks.to(pipe.target_dtype),
                action=context_action.to(pipe.target_dtype),
                kv_cache=pipe._kv_cache,
                cache_vision=True,
                rope_temporal_size=context_input.shape[2],
                start_rope_start_idx=0,
            )
            if do_cfg:
                pipe._kv_cache_neg = pipe.transformer(
                    bi_inference=False,
                    ar_txt_inference=False,
                    ar_vision_inference=True,
                    hidden_states=context_input,
                    timestep=context_timestep,
                    timestep_r=None,
                    mask_type=state.task_type,
                    return_dict=False,
                    viewmats=context_viewmats.to(pipe.target_dtype),
                    Ks=context_Ks.to(pipe.target_dtype),
                    action=context_action.to(pipe.target_dtype),
                    kv_cache=pipe._kv_cache_neg,
                    cache_vision=True,
                    rope_temporal_size=context_input.shape[2],
                    start_rope_start_idx=0,
                )

    # Denoise this chunk (same as ar_rollout inner loop)
    timesteps = state.timesteps
    with torch.no_grad():
        for t in timesteps:
            timestep_input = torch.full((CHUNK_LATENT_FRAMES,), t, device=device, dtype=timesteps.dtype)

            latent_model_input = state.latents[:, :, start_idx:end_idx]
            cond_latents_input = state.cond_latents[:, :, start_idx:end_idx]

            viewmats_input = viewmats[:, start_idx:end_idx]
            Ks_input = Ks[:, start_idx:end_idx]
            action_input = action[:, start_idx:end_idx]

            latents_concat = torch.concat([latent_model_input, cond_latents_input], dim=1)
            latents_concat = pipe.scheduler.scale_model_input(latents_concat, t)

            with torch.autocast(device_type="cuda", dtype=pipe.target_dtype, enabled=pipe.autocast_enabled):
                noise_pred_text = pipe.transformer(
                    bi_inference=False,
                    ar_txt_inference=False,
                    ar_vision_inference=True,
                    hidden_states=latents_concat,
                    timestep=timestep_input,
                    timestep_r=None,
                    mask_type=state.task_type,
                    return_dict=False,
                    viewmats=viewmats_input.to(pipe.target_dtype),
                    Ks=Ks_input.to(pipe.target_dtype),
                    action=action_input.to(pipe.target_dtype),
                    kv_cache=pipe._kv_cache,
                    cache_vision=False,
                    rope_temporal_size=latents_concat.shape[2] + len(selected_frame_indices),
                    start_rope_start_idx=len(selected_frame_indices),
                )[0]

                if do_cfg:
                    noise_pred_uncond = pipe.transformer(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        hidden_states=latents_concat,
                        timestep=timestep_input,
                        timestep_r=None,
                        mask_type=state.task_type,
                        return_dict=False,
                        viewmats=viewmats_input.to(pipe.target_dtype),
                        Ks=Ks_input.to(pipe.target_dtype),
                        action=action_input.to(pipe.target_dtype),
                        kv_cache=pipe._kv_cache_neg,
                        cache_vision=False,
                        rope_temporal_size=latents_concat.shape[2] + len(selected_frame_indices),
                        start_rope_start_idx=len(selected_frame_indices),
                    )[0]
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if pipe.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_rescale
                        )
                else:
                    noise_pred = noise_pred_text

            latent_model_input = pipe.scheduler.step(noise_pred, t, latent_model_input, return_dict=False)[0]
            state.latents[:, :, start_idx:end_idx] = latent_model_input[:, :, -CHUNK_LATENT_FRAMES:]

    # Decode NEW frames only
    frames = decode_new_frames(state, start_idx=start_idx, end_idx=end_idx, height=height, width=width)
    return frames


def decode_new_frames(state: StreamState, start_idx: int, end_idx: int, height: int, width: int) -> np.ndarray:
    pipe = state.pipe
    device = state.device

    if start_idx == 0:
        # first chunk (4 latents => 13 frames)
        latents_to_decode = state.latents[:, :, start_idx:end_idx]
        drop_first = 0
    else:
        # decode overlap one latent to get correct boundary frames; then drop the first frame (already shown)
        latents_to_decode = state.latents[:, :, start_idx - 1:end_idx]
        drop_first = 1

    # scale for VAE
    if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor:
        latents_vae = latents_to_decode / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        latents_vae = latents_to_decode / pipe.vae.config.scaling_factor

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipe.vae_dtype, enabled=pipe.vae_autocast_enabled):
        video = pipe.vae.decode(latents_vae, return_dict=False, generator=state.generator)[0]

    video = (video / 2 + 0.5).clamp(0, 1).float().cpu()  # [B,3,F,H,W]
    video = video[0].permute(1, 2, 3, 0).numpy()  # [F,H,W,3]
    video_u8 = (video * 255).round().clip(0, 255).astype(np.uint8)

    if drop_first:
        video_u8 = video_u8[drop_first:]

    return video_u8


# -----------------------------
# Pygame UI
# -----------------------------

def _read_controls(keys) -> Tuple[str, Optional[str]]:
    # User-requested mapping (game-like):
    # - Arrow keys: walk/strafe in the world
    # - WASD: look around (camera rotation)
    move = "none"
    if keys[pygame.K_UP]:
        move = "forward"
    elif keys[pygame.K_DOWN]:
        move = "backward"
    elif keys[pygame.K_RIGHT]:
        move = "right"
    elif keys[pygame.K_LEFT]:
        move = "left"

    turn = None
    if keys[pygame.K_a]:
        turn = "left"
    elif keys[pygame.K_d]:
        turn = "right"
    elif keys[pygame.K_w]:
        turn = "up"
    elif keys[pygame.K_s]:
        turn = "down"

    return move, turn


def main():
    model_path = "/home/faraz/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/21d6fbc69d2eee5e6f932a8ab2068bb9f28c6fc7"
    worldplay_path = "/home/faraz/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e"
    action_ckpt = os.path.join(worldplay_path, "ar_distilled_action_model/model.safetensors")
    image_path = "assets/img/1.png"
    prompt = "Photorealistic first-person view exploring a scene"

    print("=" * 60)
    print("vMonad - Interactive World Streaming")
    print("=" * 60)
    print("Loading pipeline once... (this is the slow part)")

    state, frame0 = init_stream(
        model_path=model_path,
        action_ckpt=action_ckpt,
        prompt=prompt,
        image_path=image_path,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
    )

    print("Ready.")
    print("Controls:")
    print("  Arrow keys: walk/strafe")
    print("  WASD: look around (camera)")
    print("  SPACE: start/stop continuous streaming")
    print("  P: open prompt box (Enter=apply, Esc=cancel; prefix '=' replaces base prompt)")
    print("  I: set initial image(s) (one path or comma-separated list; Enter=apply; Esc=cancel)")
    print("  G: toggle prompt strength (CFG) for stronger prompt injection (slower)")
    print("  Q: cycle quality (diffusion steps): 4 -> 6 -> 8 -> 12 (slower, sharper)")
    print("  M: cycle movement speed (slower = sharper exploration)")
    print("  V: toggle smooth display scaling (reduces moiré/web patterns)")
    print("  ?: show/hide keybind help")
    print("  R: reset stream state (keeps models loaded)")
    print("  1 rain | 2 fog | 3 smoke | 0 clear (optional presets)")
    print("  ESC: quit")

    pygame.init()
    screen = pygame.display.set_mode((VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE))
    pygame.display.set_caption("vMonad")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)

    # Reuse surfaces to reduce per-frame allocations (smoother UI).
    frame_surface = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT)).convert()
    scaled_surface = (
        pygame.Surface((VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE)).convert()
        if DISPLAY_SCALE != 1
        else None
    )

    current_frame = frame0
    reference_image_path = image_path
    reference_image_paths: List[str] = [image_path]
    base_prompt = prompt
    enabled_effects: List[str] = []
    custom_injection_text: Optional[str] = None
    prompt_editing = False
    image_editing = False
    prompt_buffer = ""
    guidance_idx = GUIDANCE_SCALES.index(state.pipe.guidance_scale) if state.pipe.guidance_scale in GUIDANCE_SCALES else 0
    streaming_was_on = False
    step_presets = [4, 6, 8, 12]
    step_idx = step_presets.index(state.num_inference_steps) if state.num_inference_steps in step_presets else 0
    move_step_idx = DEFAULT_MOVE_STEP_IDX
    use_smoothscale = USE_SMOOTHSCALE
    show_help = False

    # -----------------------------
    # Recording: save initial image(s) + full camera trajectory for batch re-rendering
    # -----------------------------
    reference_image_paths = [os.path.abspath(os.path.expanduser(p)) for p in reference_image_paths]
    reference_image_path = reference_image_paths[0]
    state.recorder = SegmentRecorder.new(
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        model_path=model_path,
        action_ckpt=action_ckpt,
        reference_image_paths=reference_image_paths,
        base_prompt=base_prompt,
        negative_prompt=state.negative_prompt_text,
        flow_shift=FLOW_SHIFT,
        seed=SEED,
        notes="Interactive capture from vMonad. Use <id>.json + <id>_img0.* to batch render with generate.py at higher steps.",
    )
    # record step 0
    state.recorder.append_step(state.c2w_list[0], state.action_list[0] if state.action_list else 0)
    state.recorder.log_event("segment_start", reason="program_start", image_paths=reference_image_paths)
    print(f"[record] recording to {REPETITIONS_DIR}/<timestamp>.json (+ _meta.json)", flush=True)

    # Real-time streaming plumbing (single-GPU approximation of the report’s async buffering)
    FRAME_QUEUE_MAX = 64  # ~2.6s at 24fps
    FRAME_QUEUE_TARGET = 32  # backpressure threshold for generator
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=FRAME_QUEUE_MAX)
    streaming_enabled = threading.Event()  # toggled by SPACE
    stop_event = threading.Event()
    gpu_lock = threading.Lock()  # serialize any GPU work (generation, prompt injection, reset)
    control_lock = threading.Lock()
    epoch_lock = threading.Lock()
    epoch = 0
    last_gen_time_s: float = 0.0
    gen_in_progress = False
    last_gen_err: Optional[str] = None
    last_gen_tb: Optional[str] = None
    desired_move: str = "forward"
    desired_turn: Optional[str] = None

    def _bump_epoch() -> int:
        nonlocal epoch
        with epoch_lock:
            epoch += 1
            return epoch

    def _get_epoch() -> int:
        with epoch_lock:
            return epoch

    def _clear_frame_queue():
        try:
            while True:
                frame_queue.get_nowait()
        except queue.Empty:
            return

    def _set_desired(move: str, turn: Optional[str]) -> None:
        nonlocal desired_move, desired_turn
        with control_lock:
            desired_move = move
            desired_turn = turn

    def _get_desired() -> Tuple[str, Optional[str]]:
        with control_lock:
            return desired_move, desired_turn

    def _apply_new_reference_image(new_path: str) -> Optional[str]:
        """
        Swap the initial/reference image(s) without reloading models.
        You can pass a single path, or multiple paths separated by commas/semicolons.
        The first image becomes the anchor reference for i2v conditioning; all images are
        used to compute an aggregated vision embedding (mean over images) to reduce guessing.
        Resets the stream state and recomputes vision states for the new image.
        Returns an error string if failed, else None.
        """
        nonlocal reference_image_path, frame0, current_frame
        nonlocal reference_image_paths

        raw = (new_path or "").strip()
        if not raw:
            return "empty path"

        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        if not parts:
            return "empty path"

        paths: List[str] = []
        for part in parts:
            p = os.path.expanduser(part)
            if not os.path.isabs(p):
                p = os.path.abspath(p)
            if not os.path.isfile(p):
                return f"file not found: {p}"
            paths.append(p)

        # Load all images on CPU first (so we can surface errors early)
        pil_images: List[Image.Image] = []
        for p in paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                return f"failed to open image {p}: {e}"

        # Display frame at generation resolution
        ref_display = pil_images[0].resize((VIDEO_WIDTH, VIDEO_HEIGHT))
        new_frame0 = np.array(ref_display, dtype=np.uint8)
        ref_nps = [np.array(img, dtype=np.uint8) for img in pil_images]

        with gpu_lock:
            _bump_epoch()
            _clear_frame_queue()
            streaming_enabled.clear()

            # finalize current segment and start a new one (image swap = new initial condition)
            if state.recorder is not None:
                try:
                    state.recorder.log_event("segment_end", reason="image_swap")
                    pose_p, meta_p = state.recorder.save()
                    print(f"[record] saved: {pose_p} and {meta_p}", flush=True)
                except Exception as e:
                    print(f"[record] save failed on image swap: {e}", flush=True)

            reference_image_paths = paths
            reference_image_path = paths[0]  # anchor image for i2v conditioning
            frame0 = new_frame0
            current_frame = new_frame0.copy()

            # Recompute vision states for the new image (on GPU)
            latent_len = CHUNK_LATENT_FRAMES
            latent_len, latent_h, latent_w = state.pipe.get_latent_size(
                video_length=(latent_len - 1) * state.pipe.vae_temporal_compression_ratio + 1,
                height=VIDEO_HEIGHT,
                width=VIDEO_WIDTH,
            )
            latents_dummy = torch.zeros(
                (1, state.pipe.transformer.config.in_channels, latent_len, latent_h, latent_w),
                device=state.device,
                dtype=state.pipe.target_dtype,
            )

            # Aggregate multiple reference images by averaging their vision embeddings.
            # This keeps token length fixed (fast) while incorporating more initial information.
            vs_sum: Optional[torch.Tensor] = None
            for ref_np in ref_nps:
                vs = state.pipe._prepare_vision_states(ref_np, state.pipe.ideal_resolution, latents_dummy, state.device)
                if vs is None:
                    raise RuntimeError("vision encoder is not available; cannot use reference images")
                vs_single = vs[:1]  # if CFG is on, pipeline may already repeat to batch=2
                vs_sum = vs_single if vs_sum is None else (vs_sum + vs_single)
            vision_states_single = vs_sum / float(len(ref_nps))

            state.vision_states_single = vision_states_single
            state.vision_states = (
                vision_states_single.repeat(2, 1, 1) if state.pipe.do_classifier_free_guidance else vision_states_single
            )

            # Reset stream state (new reference)
            state.c2w_list = [np.eye(4, dtype=np.float32)]
            state.w2c_list = [np.eye(4, dtype=np.float32)]
            state.K_list = [normalized_K(VIDEO_WIDTH, VIDEO_HEIGHT)]
            state.action_list = [0]
            state.latents = None
            state.cond_latents = None
            _cache_text_once(state)

            # new segment recorder (fresh initial image)
            state.recorder = SegmentRecorder.new(
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT,
                model_path=model_path,
                action_ckpt=action_ckpt,
                reference_image_paths=reference_image_paths,
                base_prompt=base_prompt,
                negative_prompt=state.negative_prompt_text,
                flow_shift=FLOW_SHIFT,
                seed=SEED,
                notes="Interactive capture from vMonad (new reference image).",
            )
            state.recorder.append_step(state.c2w_list[0], 0)
            state.recorder.log_event("segment_start", reason="image_swap", image_paths=reference_image_paths)

        return None

    def _prompt_label() -> str:
        base_snip = base_prompt.strip().replace("\n", " ")
        if len(base_snip) > 36:
            base_snip = base_snip[:33] + "..."
        if custom_injection_text:
            s = custom_injection_text.strip().replace("\n", " ")
            if len(s) > 48:
                s = s[:45] + "..."
            return f"custom: {s} | cfg={state.pipe.guidance_scale:.1f}"
        if enabled_effects:
            return "effects: " + ", ".join(sorted(enabled_effects)) + f" | cfg={state.pipe.guidance_scale:.1f}"
        return f"base: {base_snip} | cfg={state.pipe.guidance_scale:.1f}"

    def blit(frame: np.ndarray, overlay: str, input_line: Optional[str] = None):
        pygame.surfarray.blit_array(frame_surface, frame.swapaxes(0, 1))
        if DISPLAY_SCALE == 1:
            screen.blit(frame_surface, (0, 0))
        else:
            if use_smoothscale:
                pygame.transform.smoothscale(
                    frame_surface, (VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE), scaled_surface
                )
            else:
                pygame.transform.scale(
                    frame_surface, (VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE), scaled_surface
                )
            screen.blit(scaled_surface, (0, 0))
        if overlay:
            txt = font.render(overlay, True, (255, 255, 255))
            bg = pygame.Surface((txt.get_width() + 12, txt.get_height() + 8))
            bg.fill((0, 0, 0))
            bg.set_alpha(180)
            screen.blit(bg, (8, 8))
            screen.blit(txt, (14, 12))

        if input_line is not None:
            line = input_line
            # simple truncation to keep render fast + readable
            if len(line) > 140:
                line = "..." + line[-137:]
            prefix = "IMAGE> " if image_editing else "PROMPT> "
            prompt_txt = font.render(prefix + line, True, (255, 255, 255))
            bar_h = prompt_txt.get_height() + 16
            bar_w = VIDEO_WIDTH * DISPLAY_SCALE
            bar_y = VIDEO_HEIGHT * DISPLAY_SCALE - bar_h
            bar = pygame.Surface((bar_w, bar_h))
            bar.fill((0, 0, 0))
            bar.set_alpha(210)
            screen.blit(bar, (0, bar_y))
            screen.blit(prompt_txt, (10, bar_y + 8))

        # Help overlay (toggle with '?')
        if show_help:
            help_lines = [
                "vMonad Controls (? to close)",
                "",
                "Arrow keys: walk/strafe",
                "WASD: look around (camera)",
                "SPACE: start/stop continuous streaming",
                "P: prompt box (Enter=apply, Esc=cancel; prefix '=' replaces base prompt)",
                "I: change initial image(s) (comma-separated paths)",
                "G: toggle CFG strength",
                "Q: cycle diffusion steps (quality vs speed)",
                "M: cycle movement speed (slower = sharper)",
                "V: toggle smooth UI scaling (moiré reduction)",
                "R: reset stream state (models stay loaded)",
                "1 rain | 2 fog | 3 smoke | 0 clear presets",
                "ESC: quit (auto-saves recording)",
                "",
                "Recording:",
                "repetitions/<timestamp>.json + _meta.json + _img0.*",
                "Re-render: ./rerender.sh <timestamp> --steps 12 --few_step false",
            ]
            pad = 14
            line_h = font.get_linesize()
            box_w = int(VIDEO_WIDTH * DISPLAY_SCALE * 0.78)
            box_h = pad * 2 + line_h * len(help_lines)
            box = pygame.Surface((box_w, box_h))
            box.fill((0, 0, 0))
            box.set_alpha(210)
            x = 12
            y = 52  # below the status overlay
            screen.blit(box, (x, y))
            ty = y + pad
            for ln in help_lines:
                txt = font.render(ln, True, (255, 255, 255))
                screen.blit(txt, (x + pad, ty))
                ty += line_h
        pygame.display.flip()

    blit(current_frame, "Ready: SPACE=start/stop | P=prompt | G=CFG | ?=help | 1/2/3 effects | 0 clear")

    # Background generator: generates next chunks and appends frames into the queue.
    def _gen_worker():
        nonlocal last_gen_time_s, gen_in_progress, last_gen_err, last_gen_tb
        while not stop_event.is_set():
            if not streaming_enabled.is_set():
                time.sleep(0.01)
                continue
            # backpressure to avoid building huge latency
            if frame_queue.qsize() >= FRAME_QUEUE_TARGET:
                time.sleep(0.005)
                continue

            move, turn = _get_desired()
            epoch_snapshot = _get_epoch()

            t0 = time.time()
            gen_in_progress = True
            try:
                with gpu_lock:
                    frames = generate_next_chunk(
                        state,
                        move=move,
                        turn=turn,
                        reference_image_path=reference_image_path,
                        height=VIDEO_HEIGHT,
                        width=VIDEO_WIDTH,
                    )
            except Exception as e:
                last_gen_err = f"{type(e).__name__}: {e}"
                last_gen_tb = traceback.format_exc(limit=50)
                # Also print full traceback to console for debugging.
                print("[gen_worker] ERROR\n" + last_gen_tb, flush=True)
                gen_in_progress = False
                time.sleep(0.05)
                continue
            finally:
                dt = time.time() - t0
                last_gen_time_s = dt
                gen_in_progress = False

            # If prompt/reset happened mid-generation, drop stale frames.
            if epoch_snapshot != _get_epoch():
                continue

            # Stream frames into queue (drop oldest if we fall behind)
            for fr in frames:
                if stop_event.is_set():
                    break
                if epoch_snapshot != _get_epoch():
                    break
                try:
                    frame_queue.put(fr, timeout=0.1)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

    gen_thread = threading.Thread(target=_gen_worker, name="worldplay-gen", daemon=True)
    gen_thread.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save recording on window close
                if state.recorder is not None:
                    try:
                        state.recorder.log_event("segment_end", reason="window_close")
                        pose_p, meta_p = state.recorder.save()
                        print(f"[record] saved: {pose_p} and {meta_p}", flush=True)
                    except Exception as e:
                        print(f"[record] save failed on window close: {e}", flush=True)
                running = False
            elif event.type == pygame.KEYDOWN:
                # Prompt input mode: capture keystrokes into a textbox.
                if prompt_editing or image_editing:
                    if event.key == pygame.K_ESCAPE:
                        prompt_editing = False
                        image_editing = False
                        prompt_buffer = ""
                        pygame.key.stop_text_input()
                        # restore streaming state (if it was running)
                        if streaming_was_on:
                            streaming_enabled.set()
                        blit(current_frame, "Input cancelled", None)
                        continue
                    if event.key == pygame.K_RETURN:
                        text = prompt_buffer.strip()
                        was_prompt = prompt_editing
                        prompt_editing = False
                        image_editing = False
                        prompt_buffer = ""
                        pygame.key.stop_text_input()

                        if was_prompt:
                            # restore streaming state (if it was running)
                            if streaming_was_on:
                                streaming_enabled.set()

                            if text:
                                # Prefix '=' to replace base prompt (full prompt change).
                                # Otherwise treat input as an injection suffix to preserve scene continuity.
                                if text.startswith("="):
                                    base_prompt = text[1:].strip()
                                    enabled_effects = []
                                    custom_injection_text = None
                                    blit(current_frame, "Applying base prompt...", None)
                                    pygame.event.pump()
                                    with gpu_lock:
                                        _bump_epoch()
                                        _clear_frame_queue()
                                        apply_prompt_injection(state, base_prompt, negative_prompt="")
                                    blit(current_frame, "Base prompt updated", None)
                                else:
                                    enabled_effects = []
                                    custom_injection_text = text
                                    injected, effects_all = build_injected_prompt(
                                        base_prompt, enabled_effects, custom_injection_text
                                    )
                                    negative_prompt = build_negative_prompt(effects_all)
                                    blit(current_frame, "Applying injected prompt...", None)
                                    pygame.event.pump()
                                    with gpu_lock:
                                        _bump_epoch()
                                        _clear_frame_queue()
                                        apply_prompt_injection(state, injected, negative_prompt=negative_prompt)
                                    blit(current_frame, f"Prompt active: {_prompt_label()}", None)
                            else:
                                blit(current_frame, f"Prompt active: {_prompt_label()}", None)
                        else:
                            # Image path entry: do NOT auto-resume streaming; image swap resets stream state.
                            if text:
                                blit(current_frame, "Loading image...", None)
                                err = _apply_new_reference_image(text)
                                if err:
                                    blit(current_frame, f"Image error: {err}", None)
                                else:
                                    blit(current_frame, "Image updated. Press SPACE to start.", None)
                            else:
                                blit(current_frame, "Image input empty", None)
                        continue

                    if event.key == pygame.K_BACKSPACE:
                        prompt_buffer = prompt_buffer[:-1]
                    elif event.key == pygame.K_DELETE:
                        prompt_buffer = ""
                    else:
                        ch = event.unicode
                        if ch and ch.isprintable():
                            prompt_buffer += ch

                    blit(
                        current_frame,
                        (
                            "Type prompt. Enter=apply, Esc=cancel. Prefix '=' replaces base prompt."
                            if prompt_editing
                            else "Type image path(s). Use commas for multiple. Enter=apply, Esc=cancel."
                        ),
                        prompt_buffer,
                    )
                    continue

                if event.key == pygame.K_ESCAPE:
                    # Save recording on exit
                    if state.recorder is not None:
                        try:
                            state.recorder.log_event("segment_end", reason="exit")
                            pose_p, meta_p = state.recorder.save()
                            print(f"[record] saved: {pose_p} and {meta_p}", flush=True)
                        except Exception as e:
                            print(f"[record] save failed on exit: {e}", flush=True)
                    running = False
                elif event.unicode == "?" or (
                    event.key == pygame.K_SLASH and (pygame.key.get_mods() & pygame.KMOD_SHIFT)
                ):
                    show_help = not show_help
                    blit(current_frame, "Help shown" if show_help else "Help hidden", None)
                elif event.key == pygame.K_p:
                    # Pause streaming while typing to avoid accidental movement and to apply injection cleanly.
                    streaming_was_on = streaming_enabled.is_set()
                    streaming_enabled.clear()
                    prompt_editing = True
                    image_editing = False
                    prompt_buffer = ""
                    pygame.key.start_text_input()
                    blit(
                        current_frame,
                        "Type prompt. Enter=apply, Esc=cancel. Prefix '=' replaces base prompt.",
                        prompt_buffer,
                    )
                elif event.key == pygame.K_i:
                    # Change initial image (path input)
                    streaming_was_on = streaming_enabled.is_set()
                    streaming_enabled.clear()
                    image_editing = True
                    prompt_editing = False
                    prompt_buffer = ""
                    pygame.key.start_text_input()
                    blit(
                        current_frame,
                        "Type image path(s). Use commas for multiple. Enter=apply, Esc=cancel.",
                        prompt_buffer,
                    )
                elif event.key == pygame.K_g:
                    # Toggle CFG guidance scale (affects prompt injection strength)
                    with gpu_lock:
                        _bump_epoch()
                        _clear_frame_queue()
                        new_idx = (guidance_idx + 1) % len(GUIDANCE_SCALES)
                        state.pipe._guidance_scale = GUIDANCE_SCALES[new_idx]
                        state.pipe._guidance_rescale = GUIDANCE_RESCALE
                        apply_prompt_injection(
                            state,
                            state.prompt_text,
                            negative_prompt=state.negative_prompt_text,
                        )
                        guidance_idx = new_idx
                    blit(current_frame, f"CFG scale: {state.pipe.guidance_scale:.1f}", None)
                    blit(current_frame, f"Prompt active: {_prompt_label()}", None)
                    if state.recorder is not None:
                        state.recorder.log_event("toggle_cfg", guidance_scale=state.pipe.guidance_scale)
                elif event.key == pygame.K_q:
                    # Cycle diffusion steps for quality vs speed.
                    # Keep this inside the GPU lock so it can't flip mid-chunk in a way that surprises the scheduler.
                    with gpu_lock:
                        _bump_epoch()
                        _clear_frame_queue()
                        step_idx = (step_idx + 1) % len(step_presets)
                        state.num_inference_steps = step_presets[step_idx]
                    blit(current_frame, f"Steps per chunk: {state.num_inference_steps}", None)
                    if state.recorder is not None:
                        state.recorder.log_event("toggle_steps", num_inference_steps=state.num_inference_steps)
                elif event.key == pygame.K_m:
                    # Slower movement keeps the camera closer to known regions -> less blur.
                    move_step_idx = (move_step_idx + 1) % len(MOVE_STEP_PRESETS)
                    globals()["DEFAULT_MOVE_STEP_IDX"] = move_step_idx
                    blit(current_frame, f"Move step: {MOVE_STEP_PRESETS[move_step_idx]:.2f}", None)
                    if state.recorder is not None:
                        state.recorder.log_event("toggle_move_step", move_step=MOVE_STEP_PRESETS[move_step_idx])
                elif event.key == pygame.K_v:
                    use_smoothscale = not use_smoothscale
                    blit(current_frame, f"Smoothscale: {'ON' if use_smoothscale else 'OFF'}", None)
                    if state.recorder is not None:
                        state.recorder.log_event("toggle_ui_smoothscale", enabled=use_smoothscale)
                elif event.key == pygame.K_SPACE:
                    # Toggle continuous streaming
                    if streaming_enabled.is_set():
                        streaming_enabled.clear()
                        blit(current_frame, "Streaming paused", None)
                    else:
                        streaming_enabled.set()
                        blit(current_frame, "Streaming running", None)
                elif event.key == pygame.K_r:
                    # Reset stream state but keep models
                    if DEBUG:
                        print("[reset] resetting streaming state")
                    with gpu_lock:
                        _bump_epoch()
                        _clear_frame_queue()
                        streaming_enabled.clear()
                        # finalize current segment and start a new one (reset = new trajectory from same initial)
                        if state.recorder is not None:
                            try:
                                state.recorder.log_event("segment_end", reason="reset")
                                pose_p, meta_p = state.recorder.save()
                                print(f"[record] saved: {pose_p} and {meta_p}", flush=True)
                            except Exception as e:
                                print(f"[record] save failed on reset: {e}", flush=True)
                        state.c2w_list = [np.eye(4, dtype=np.float32)]
                        state.w2c_list = [np.eye(4, dtype=np.float32)]
                        state.K_list = [normalized_K(VIDEO_WIDTH, VIDEO_HEIGHT)]
                        state.action_list = [0]
                        state.latents = None
                        state.cond_latents = None
                        _cache_text_once(state)
                    current_frame = frame0.copy()
                    blit(current_frame, "RESET (press SPACE to resume)", None)
                    state.recorder = SegmentRecorder.new(
                        width=VIDEO_WIDTH,
                        height=VIDEO_HEIGHT,
                        model_path=model_path,
                        action_ckpt=action_ckpt,
                        reference_image_paths=reference_image_paths,
                        base_prompt=base_prompt,
                        negative_prompt=state.negative_prompt_text,
                        flow_shift=FLOW_SHIFT,
                        seed=SEED,
                        notes="Interactive capture from vMonad (reset).",
                    )
                    state.recorder.append_step(state.c2w_list[0], 0)
                    state.recorder.log_event("segment_start", reason="reset", image_paths=reference_image_paths)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_0):
                    # Prompt/effect injection (applied immediately, between chunks)
                    if event.key == pygame.K_0:
                        enabled_effects = []
                        custom_injection_text = None
                    else:
                        custom_injection_text = None
                        key_to_effect = {
                            pygame.K_1: "rain",
                            pygame.K_2: "fog",
                            pygame.K_3: "smoke",
                        }
                        eff = key_to_effect.get(event.key)
                        if eff:
                            if eff in enabled_effects:
                                enabled_effects = [x for x in enabled_effects if x != eff]
                            else:
                                enabled_effects = enabled_effects + [eff]

                    injected, effects_all = build_injected_prompt(base_prompt, enabled_effects, None)
                    negative_prompt = build_negative_prompt(effects_all)
                    blit(current_frame, f"Applying prompt: {', '.join(effects_all) or 'none'}")
                    pygame.event.pump()
                    with gpu_lock:
                        _bump_epoch()
                        _clear_frame_queue()
                        apply_prompt_injection(state, injected, negative_prompt=negative_prompt)
                    blit(current_frame, f"Prompt active: {_prompt_label()}", None)
        # Update desired controls continuously (only when not typing in prompt box)
        if not (prompt_editing or image_editing):
            keys = pygame.key.get_pressed()
            move, turn = _read_controls(keys)
            _set_desired(move, turn)

        # Pull a frame from queue at playback rate
        try:
            current_frame = frame_queue.get_nowait()
        except queue.Empty:
            pass

        status = "RUN" if streaming_enabled.is_set() else "PAUSE"
        qn = frame_queue.qsize()
        gen_tag = "gen..." if gen_in_progress else f"gen {last_gen_time_s:.2f}s"
        steps_tag = f"steps={getattr(state, 'num_inference_steps', NUM_INFERENCE_STEPS)}"
        move_tag = f"mv={MOVE_STEP_PRESETS[DEFAULT_MOVE_STEP_IDX]:.2f}"
        scale_tag = "ui=smooth" if (DISPLAY_SCALE != 1 and use_smoothscale) else ("ui=fast" if DISPLAY_SCALE != 1 else "ui=1x")
        if last_gen_err:
            overlay = f"{status} | {_prompt_label()} | {steps_tag} | {move_tag} | {scale_tag} | q={qn} | ERR: {last_gen_err}"
        else:
            overlay = f"{status} | {_prompt_label()} | {steps_tag} | {move_tag} | {scale_tag} | q={qn} | {gen_tag}"
        blit(current_frame, overlay, prompt_buffer if (prompt_editing or image_editing) else None)

        # UI tick: 24fps during playback/streaming, otherwise faster for responsiveness
        if streaming_enabled.is_set() or qn > 0:
            clock.tick(FPS_PLAYBACK)
        else:
            clock.tick(60)

    pygame.quit()
    stop_event.set()
    try:
        gen_thread.join(timeout=1.0)
    except Exception:
        pass


if __name__ == "__main__":
    main()
