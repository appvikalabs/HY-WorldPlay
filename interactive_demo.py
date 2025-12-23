#!/usr/bin/env python3
"""
HY-WorldPlay Interactive Demo - Simple approach using the working pipeline.

Loads model once, then generates short clips based on keyboard input.
"""

import os
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import pygame
from PIL import Image
import json
import tempfile
from scipy.spatial.transform import Rotation as R

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons import is_flash_available
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

# Initialize
parallel_dims = initialize_parallel_state(sp=1)
torch.cuda.set_device(0)

class Args:
    attn_mode = "torch"
    ssta = False
    enable_torch_compile = False

initialize_infer_state(Args())


# Resolution settings - lower = faster
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 368  # Must be divisible by 16

def create_trajectory_from_pose(num_latent_frames, start_position, start_rotation, move_dir="forward", turn_dir=None):
    """Create camera trajectory starting from accumulated pose.
    
    Args:
        num_latent_frames: Number of poses to generate
        start_position: [x, y, z] starting position
        start_rotation: [yaw, pitch, roll] in degrees
        move_dir: Movement direction
        turn_dir: Rotation direction
    
    Returns:
        poses: Dictionary of poses
        end_position: Final position after trajectory
        end_rotation: Final rotation after trajectory
    """
    poses = {}
    
    # Base camera setup - 3x3 intrinsic matrix
    fx = fy = VIDEO_WIDTH * 1.01
    cx, cy = VIDEO_WIDTH, VIDEO_HEIGHT
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Movement per frame
    move_per_frame = 0.08
    rot_per_frame = 5.0  # degrees
    
    current_pos = np.array(start_position, dtype=float)
    current_rot = np.array(start_rotation, dtype=float)  # [yaw, pitch, roll]
    
    for i in range(num_latent_frames):
        # Calculate movement delta based on current yaw (for proper direction)
        yaw_rad = np.radians(current_rot[0])
        
        # Movement in world space based on facing direction
        if move_dir == "forward":
            delta = np.array([np.sin(yaw_rad), 0, np.cos(yaw_rad)]) * move_per_frame
        elif move_dir == "backward":
            delta = np.array([-np.sin(yaw_rad), 0, -np.cos(yaw_rad)]) * move_per_frame
        elif move_dir == "left":
            delta = np.array([-np.cos(yaw_rad), 0, np.sin(yaw_rad)]) * move_per_frame
        elif move_dir == "right":
            delta = np.array([np.cos(yaw_rad), 0, -np.sin(yaw_rad)]) * move_per_frame
        else:
            delta = np.array([0, 0, 0])
        
        # Apply movement
        if i > 0:  # Don't move on first frame (matches working example)
            current_pos += delta
        
        # Rotation delta
        if turn_dir == "left":
            current_rot[0] -= rot_per_frame if i > 0 else 0
        elif turn_dir == "right":
            current_rot[0] += rot_per_frame if i > 0 else 0
        elif turn_dir == "up":
            current_rot[1] -= rot_per_frame * 0.7 if i > 0 else 0
        elif turn_dir == "down":
            current_rot[1] += rot_per_frame * 0.7 if i > 0 else 0
        
        # Clamp pitch
        current_rot[1] = np.clip(current_rot[1], -80, 80)
        
        # Build rotation matrix
        rot = R.from_euler('yxz', [current_rot[0], current_rot[1], current_rot[2]], degrees=True)
        
        # Camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, :3] = rot.as_matrix()
        c2w[:3, 3] = current_pos
        
        poses[str(i)] = {
            "extrinsic": c2w.tolist(),
            "K": K.tolist()
        }
    
    return poses, current_pos.tolist(), current_rot.tolist()


def main():
    # Paths
    model_path = "/home/faraz/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/21d6fbc69d2eee5e6f932a8ab2068bb9f28c6fc7"
    worldplay_path = "/home/faraz/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e"
    action_ckpt = os.path.join(worldplay_path, "ar_distilled_action_model/model.safetensors")
    image_path = "assets/img/1.png"
    
    print("="*60)
    print("HY-WorldPlay Interactive Demo")
    print("="*60)
    print("Loading model (one-time cost)...")
    
    # Load pipeline once
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

    # Prefer PyTorch SDPA when the external `flash-attn` package isn't installed.
    if not is_flash_available() and hasattr(pipe, "transformer") and pipe.transformer is not None:
        pipe.transformer.set_attn_mode("torch")
    
    print("Model loaded!")
    print()
    print("Controls:")
    print("  W/S - Move forward/backward")
    print("  A/D - Strafe left/right")
    print("  Arrow keys - Look around")
    print("  SPACE - Generate video for current movement")
    print("  R - Reset to initial image")
    print("  ESC - Exit")
    print("="*60)
    
    # Initialize pygame - display at 2x for visibility
    DISPLAY_SCALE = 2
    pygame.init()
    screen = pygame.display.set_mode((VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE))
    pygame.display.set_caption("HY-WorldPlay Interactive (Optimized)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Load and display initial image at generation resolution
    ref_img = Image.open(image_path).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT))
    current_frame = np.array(ref_img)
    original_image_path = image_path  # ALWAYS use original for reference
    
    # Consistent seed for temporal coherence
    base_seed = 42
    
    # Cumulative camera pose (persists across generations)
    camera_position = [0.0, 0.0, 0.0]  # x, y, z
    camera_rotation = [0.0, 0.0, 0.0]  # yaw, pitch, roll in degrees
    
    # State
    move_dir = None
    turn_dir = None
    pending_move = None  # Captured when SPACE pressed
    pending_turn = None
    generating = False
    
    def display_frame(frame, text=""):
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Scale up for display using pygame transform (faster)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (VIDEO_WIDTH * DISPLAY_SCALE, VIDEO_HEIGHT * DISPLAY_SCALE))
        screen.blit(surf, (0, 0))
        
        if text:
            txt = font.render(text, True, (255, 255, 255))
            bg = pygame.Surface((txt.get_width() + 10, txt.get_height() + 10))
            bg.fill((0, 0, 0))
            bg.set_alpha(180)
            screen.blit(bg, (5, 5))
            screen.blit(txt, (10, 10))
        pygame.display.flip()
    
    display_frame(current_frame, "Ready - Press WASD to move, Arrows to look")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    move_dir = "forward"
                elif event.key == pygame.K_s:
                    move_dir = "backward"
                elif event.key == pygame.K_a:
                    move_dir = "left"
                elif event.key == pygame.K_d:
                    move_dir = "right"
                elif event.key == pygame.K_LEFT:
                    turn_dir = "left"
                elif event.key == pygame.K_RIGHT:
                    turn_dir = "right"
                elif event.key == pygame.K_UP:
                    turn_dir = "up"
                elif event.key == pygame.K_DOWN:
                    turn_dir = "down"
                elif event.key == pygame.K_SPACE:
                    if move_dir or turn_dir:
                        pending_move = move_dir
                        pending_turn = turn_dir
                        generating = True
                elif event.key == pygame.K_r:
                    # Reset to initial image and pose
                    print("Resetting to initial image and pose...")
                    ref_img = Image.open(image_path).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                    current_frame = np.array(ref_img)
                    camera_position = [0.0, 0.0, 0.0]
                    camera_rotation = [0.0, 0.0, 0.0]
                    display_frame(current_frame, "RESET - Ready")
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    move_dir = None
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                    turn_dir = None
        
        # Update display with current state
        status = []
        if move_dir:
            status.append(f"Move: {move_dir}")
        if turn_dir:
            status.append(f"Turn: {turn_dir}")
        if status:
            status.append("Press SPACE to generate")
        else:
            status.append("Use WASD/Arrows, SPACE to generate")
        
        display_frame(current_frame, " | ".join(status))
        
        # Generate if requested
        if generating and (pending_move or pending_turn):
            display_frame(current_frame, "Generating...")
            pygame.display.flip()
            
            # DEBUG: Print camera state BEFORE generation
            print(f"\n=== GENERATION START ===")
            print(f"  camera_position BEFORE: {camera_position}")
            print(f"  camera_rotation BEFORE: {camera_rotation}")
            print(f"  pending_move: {pending_move}, pending_turn: {pending_turn}")
            
            # Create trajectory using captured directions and cumulative pose
            # Video frames must satisfy: ((num_frames - 1) // 4 + 1) % 4 == 0
            num_frames = 13  # (13-1)//4 + 1 = 4 latent frames, divisible by 4 ✓
            latent_chunk_num = (num_frames - 1) // 4 + 1  # = 4
            poses, new_pos, new_rot = create_trajectory_from_pose(
                latent_chunk_num, 
                camera_position, 
                camera_rotation,
                pending_move or "forward", 
                pending_turn
            )
            
            # DEBUG: Print trajectory output
            print(f"  new_pos from trajectory: {new_pos}")
            print(f"  new_rot from trajectory: {new_rot}")
            
            # Save trajectory to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(poses, f)
                pose_path = f.name
            
            try:
                # Import pose processing from generate.py
                from generate import pose_to_input
                viewmats, Ks, action = pose_to_input(pose_path, latent_chunk_num)
                print(f"Actions: {action.tolist()}")  # Debug: show action codes
                
                # Generate with optimized settings
                # ALWAYS use original reference image for quality
                out = pipe(
                    enable_sr=False,
                    prompt="Photorealistic first-person view exploring a scene",
                    aspect_ratio="auto",
                    num_inference_steps=4,  # Distilled model - fast
                    sr_num_inference_steps=None,
                    video_length=num_frames,
                    negative_prompt="",
                    seed=base_seed,  # Consistent seed for coherence
                    output_type="pt",
                    prompt_rewrite=False,
                    return_pre_sr_video=False,
                    viewmats=viewmats.unsqueeze(0),
                    Ks=Ks.unsqueeze(0),
                    action=action.unsqueeze(0),
                    few_step=True,
                    chunk_latent_frames=4,
                    model_type="ar",
                    reference_image=original_image_path,  # ALWAYS original
                    user_height=VIDEO_HEIGHT,
                    user_width=VIDEO_WIDTH,
                )
                
                # Play frames
                video = out.videos[0]  # C, T, H, W
                print(f"Video tensor shape: {video.shape}")
                video = (video * 255).clamp(0, 255).byte()
                video = video.permute(1, 2, 3, 0).cpu().numpy()  # T, H, W, C
                print(f"Video numpy shape: {video.shape}")
                
                for i, frame in enumerate(video):
                    print(f"Displaying frame {i+1}/{len(video)}, shape: {frame.shape}")
                    display_frame(frame, f"Playing {i+1}/{len(video)}...")
                    pygame.event.pump()  # Keep display responsive
                    pygame.time.wait(42)  # ~24 fps
                    
                    # Check for escape
                    for ev in pygame.event.get():
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            running = False
                
                # Keep last frame for display
                current_frame = video[-1]
                
                # DEBUG: Print before update
                print(f"  BEFORE update: camera_position = {camera_position}")
                
                # Update cumulative camera pose for next generation
                camera_position = new_pos
                camera_rotation = new_rot
                
                # DEBUG: Print after update
                print(f"  AFTER update: camera_position = {camera_position}")
                print(f"  AFTER update: camera_rotation = {camera_rotation}")
                print(f"=== GENERATION END ===\n")
                
            except Exception as e:
                print(f"Generation error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                os.unlink(pose_path)
            
            generating = False
            pending_move = None
            pending_turn = None
        
        clock.tick(30)
    
    pygame.quit()


if __name__ == "__main__":
    main()
