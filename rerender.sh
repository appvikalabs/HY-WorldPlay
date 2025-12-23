#!/bin/bash
# vMonad / HY-WorldPlay: Re-render a recorded interactive session with higher quality
# -------------------------------------------------------------------------------
# This script consumes the files produced by interactive_streaming_world.py:
#   repetitions/<id>.json
#   repetitions/<id>_meta.json
#   repetitions/<id>_img0.(png/jpg/...)
#
# Examples:
#   ./rerender.sh 20251223-153012 --steps 12 --few_step false
#   ./rerender.sh repetitions/20251223-153012_meta.json --steps 12 --prompt "cinematic, sharp"
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <segment_id | repetitions/<id>_meta.json> [--steps N] [--few_step true|false] [--prompt STR] [--negative STR] [--model_path PATH] [--action_ckpt PATH] [--image PATH] [--out PATH] [--env ENVNAME]"
  exit 2
fi

SEG_OR_META="$1"
shift || true

# Defaults / overrides
STEPS=""
FEW_STEP=""
PROMPT_OVERRIDE=""
NEG_OVERRIDE=""
MODEL_PATH_OVERRIDE=""
ACTION_CKPT_OVERRIDE=""
IMAGE_OVERRIDE=""
OUT_OVERRIDE=""
ENVNAME="${WORLDPLAY_ENV:-worldplay312}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="${2:-}"; shift 2 ;;
    --few_step) FEW_STEP="${2:-}"; shift 2 ;;
    --prompt) PROMPT_OVERRIDE="${2:-}"; shift 2 ;;
    --negative) NEG_OVERRIDE="${2:-}"; shift 2 ;;
    --model_path) MODEL_PATH_OVERRIDE="${2:-}"; shift 2 ;;
    --action_ckpt) ACTION_CKPT_OVERRIDE="${2:-}"; shift 2 ;;
    --image) IMAGE_OVERRIDE="${2:-}"; shift 2 ;;
    --out) OUT_OVERRIDE="${2:-}"; shift 2 ;;
    --env) ENVNAME="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 <segment_id | repetitions/<id>_meta.json> [--steps N] [--few_step true|false] [--prompt STR] [--negative STR] [--model_path PATH] [--action_ckpt PATH] [--image PATH] [--out PATH] [--env ENVNAME]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

# Resolve meta path
if [[ "$SEG_OR_META" == *.json && -f "$SEG_OR_META" && "$SEG_OR_META" == *"_meta.json" ]]; then
  META_PATH="$SEG_OR_META"
else
  META_PATH="repetitions/${SEG_OR_META}_meta.json"
fi

if [[ ! -f "$META_PATH" ]]; then
  echo "Meta file not found: $META_PATH"
  exit 1
fi

# Pull derived defaults from meta via python (no jq dependency).
read -r POSE_JSON IMAGE_PATH MODEL_PATH ACTION_CKPT WIDTH HEIGHT SEED BASE_PROMPT NEG_PROMPT NUM_LATENTS <<<"$(python - <<'PY'
import json, os, sys
meta_path = sys.argv[1]
with open(meta_path, "r") as f:
    meta = json.load(f)
pose_json = meta.get("batch_render_hint", {}).get("pose_json_path") or meta.get("pose_json_path")
if not pose_json:
    sid = meta.get("segment_id")
    pose_json = os.path.join("repetitions", f"{sid}.json") if sid else ""
imgs = meta.get("copied_reference_images") or []
image_path = imgs[0] if imgs else (meta.get("reference_image_paths") or [None])[0]
model_path = meta.get("model_path") or ""
action_ckpt = meta.get("action_ckpt") or ""
w = meta.get("width") or 640
h = meta.get("height") or 368
seed = meta.get("seed") or 42
base_prompt = (meta.get("base_prompt") or "").replace("\n", " ")
neg_prompt = (meta.get("negative_prompt") or "").replace("\n", " ")
num_latents = meta.get("num_latent_steps") or 0
print(pose_json, image_path, model_path, action_ckpt, w, h, seed, base_prompt, neg_prompt, num_latents)
PY
"$META_PATH")"

if [[ -z "${POSE_JSON:-}" || ! -f "${POSE_JSON:-}" ]]; then
  echo "Pose JSON not found (from meta): ${POSE_JSON:-<empty>}"
  exit 1
fi

if [[ -z "${IMAGE_PATH:-}" || ! -f "${IMAGE_PATH:-}" ]]; then
  echo "Reference image not found (from meta): ${IMAGE_PATH:-<empty>}"
  exit 1
fi

# Apply overrides
[[ -n "$MODEL_PATH_OVERRIDE" ]] && MODEL_PATH="$MODEL_PATH_OVERRIDE"
[[ -n "$ACTION_CKPT_OVERRIDE" ]] && ACTION_CKPT="$ACTION_CKPT_OVERRIDE"
[[ -n "$IMAGE_OVERRIDE" ]] && IMAGE_PATH="$IMAGE_OVERRIDE"
[[ -n "$PROMPT_OVERRIDE" ]] && BASE_PROMPT="$PROMPT_OVERRIDE"
[[ -n "$NEG_OVERRIDE" ]] && NEG_PROMPT="$NEG_OVERRIDE"

if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH:-}" ]]; then
  echo "Model path missing or not a directory: ${MODEL_PATH:-<empty>}"
  exit 1
fi
if [[ -z "${ACTION_CKPT:-}" || ! -f "${ACTION_CKPT:-}" ]]; then
  echo "Action checkpoint missing or not a file: ${ACTION_CKPT:-<empty>}"
  exit 1
fi

if [[ -z "$STEPS" ]]; then
  STEPS="30"
fi
if [[ -z "$FEW_STEP" ]]; then
  FEW_STEP="false"
fi

# Derive video_length from recorded latent steps (generate.py expects frame length).
# video_length = (latents-1)*4+1
VIDEO_LENGTH="$(python - <<'PY'
import sys
latents = int(sys.argv[1])
if latents <= 0:
    print(13)  # fallback
else:
    print((latents - 1) * 4 + 1)
PY
"${NUM_LATENTS:-0}")"

OUT_PATH="$OUT_OVERRIDE"
if [[ -z "$OUT_PATH" ]]; then
  mkdir -p outputs/rerenders
  OUT_PATH="outputs/rerenders/$(basename "$META_PATH" "_meta.json")_steps${STEPS}.mp4"
fi

echo "Re-rendering:"
echo "  meta:      $META_PATH"
echo "  pose:      $POSE_JSON"
echo "  image:     $IMAGE_PATH"
echo "  size:      ${WIDTH}x${HEIGHT}"
echo "  latents:   ${NUM_LATENTS}"
echo "  frames:    ${VIDEO_LENGTH}"
echo "  steps:     ${STEPS}"
echo "  few_step:  ${FEW_STEP}"
echo "  out:       $OUT_PATH"
echo "  env:       $ENVNAME"
echo

source ~/anaconda3/bin/activate "$ENVNAME"

# Note: generate.py expects resolution/aspect_ratio but we also pass width/height explicitly.
torchrun --nproc_per_node=1 generate.py \
  --prompt "$BASE_PROMPT" \
  --negative_prompt "$NEG_PROMPT" \
  --image_path "$IMAGE_PATH" \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --video_length "$VIDEO_LENGTH" \
  --seed "$SEED" \
  --rewrite false \
  --sr false \
  --pose_json_path "$POSE_JSON" \
  --output_path "$(dirname "$OUT_PATH")" \
  --model_path "$MODEL_PATH" \
  --action_ckpt "$ACTION_CKPT" \
  --few_step "$FEW_STEP" \
  --num_inference_steps "$STEPS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --model_type 'ar'

echo
echo "Done. Check: $OUT_PATH (generated into $(dirname "$OUT_PATH")/, filename set by generate.py)"
