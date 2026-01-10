# 📷 HY-WorldPlay Camera Trajectory Guide

This guide documents camera control for HY-WorldPlay, based on experimentation with the model.

---

## 🎮 Understanding WorldPlay

WorldPlay is a **world simulation model**, not just a camera controller:

| Capability | Supported |
|------------|-----------|
| Camera movement (translate/rotate) | ✅ YES |
| World dynamics (objects can move/animate) | ✅ YES |
| Character animation from prompt | ✅ YES (context-dependent) |
| Independent character control | ❌ NO |

The prompt influences what happens in the world. For example:
- "girl runs toward castle" → model may animate her running
- Camera trajectory controls viewpoint, world responds naturally

---

## 📐 Coordinate System

```
                    Castle / Forward
                          ↑
                          Z+
                          
         Left             │             Right
          -X ←────────────┼────────────→ +X
                          │
                          │
                          ↓
                         -Z
                    Behind / Backward
                    
         Y+ = Down (in camera coords)
         Y- = Up
```

**Initial Setup (typical):**
- Subject at origin `(0, 0, 0)`
- Subject facing `+Z` (toward castle/forward)
- Camera behind subject at `-Z`
- Camera also facing `+Z` (looking at subject's back)

---

## 🎬 Trajectory JSON Format

```json
{
    "0": {
        "extrinsic": [
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "K": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]
    },
    "1": { ... },
    ...
}
```

| Field | Description |
|-------|-------------|
| `extrinsic` | 4×4 camera-to-world transformation matrix |
| `K` | 3×3 camera intrinsic matrix |
| `r00-r22` | 3×3 rotation matrix |
| `tx, ty, tz` | Camera position in world coordinates |
| `fx, fy` | Focal length (default: ~970) |
| `cx, cy` | Principal point (default: image center) |

---

## 🚀 Motion Types

### 1. Forward Walk (Simple)

Camera moves forward through the world.

```python
motions = [{"forward": 0.08} for _ in range(num_frames)]
```

**Effect:** First-person walking toward the scene.

---

### 2. Turn/Pan (Yaw Rotation)

Camera rotates left or right.

```python
motions = [{"yaw": np.deg2rad(3)} for _ in range(num_frames)]  # Turn right
motions = [{"yaw": np.deg2rad(-3)} for _ in range(num_frames)] # Turn left
```

**Effect:** Looking around, panning.

---

### 3. Look Up/Down (Pitch Rotation)

Camera tilts up or down.

```python
motions = [{"pitch": np.deg2rad(2)} for _ in range(num_frames)]  # Look up
motions = [{"pitch": np.deg2rad(-2)} for _ in range(num_frames)] # Look down
```

---

### 4. Orbit Around Subject

Camera circles around a subject at the origin.

```python
def generate_orbit(num_frames, radius, start_angle, end_angle, height=0.3):
    poses = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        angle = start_angle + t * (end_angle - start_angle)
        
        # Camera position on circle
        cam_x = -radius * np.sin(angle)
        cam_z = -radius * np.cos(angle)
        cam_y = height
        
        # Camera looks at origin (subject)
        # Build look-at rotation matrix
        # ... (see full code below)
        
        poses.append(T)
    return poses
```

**Key insight:** For orbits, start angle and end angle matter:
- `0°` = Camera behind subject (`-Z`)
- `90°` = Camera at subject's left (`-X`)
- `180°` = Camera in front of subject (`+Z`)
- `270°` = Camera at subject's right (`+X`)

---

## 📏 Video Length Constraints

| Model Type | Constraint | Valid Lengths |
|------------|------------|---------------|
| AR (autoregressive) | `(frames-1)//4 + 1` divisible by **4** | 61, 77, 93, 109, 125 |
| Bidirectional | `(frames-1)//4 + 1` divisible by **16** | 61, 125 |

**Duration at 24fps:**
| Frames | Latents | Duration |
|--------|---------|----------|
| 61 | 16 | 2.5 sec |
| 77 | 20 | 3.2 sec |
| 93 | 24 | 3.9 sec |
| 109 | 28 | 4.5 sec |
| 125 | 32 | 5.2 sec |

---

## 🎯 Complete Orbit Example

Quarter orbit from behind subject to their left side:

```python
import numpy as np
import json

def look_at_direction(forward_dir, up=np.array([0, 1, 0])):
    """Create rotation matrix for camera looking in forward_dir."""
    forward = forward_dir / np.linalg.norm(forward_dir)
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    up_new = np.cross(forward, right)
    
    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = up_new
    R[:, 2] = forward
    return R

def generate_quarter_orbit(num_frames, radius=2.0, height=0.3):
    """
    Quarter orbit: behind (-Z) to left side (-X)
    Subject at origin, facing +Z
    """
    poses = []
    
    start_angle = 0           # Behind (-Z)
    end_angle = np.pi / 2     # Left side (-X)
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        angle = start_angle + t * (end_angle - start_angle)
        
        # Camera position on circle
        cam_x = -radius * np.sin(angle)
        cam_z = -radius * np.cos(angle)
        cam_y = height
        
        camera_pos = np.array([cam_x, cam_y, cam_z])
        target = np.array([0, 0, 0])  # Subject at origin
        
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        R = look_at_direction(forward)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = camera_pos
        poses.append(T.copy())
    
    return poses

# Generate 32 latents = 125 frames = 5.2 seconds
poses = generate_quarter_orbit(num_frames=32, radius=2.0, height=0.3)

# Save to JSON
intrinsic = [[969.69, 0.0, 960.0], [0.0, 969.69, 540.0], [0.0, 0.0, 1.0]]
trajectory = {str(i): {"extrinsic": p.tolist(), "K": intrinsic} 
              for i, p in enumerate(poses)}

with open("./assets/pose/my_orbit.json", "w") as f:
    json.dump(trajectory, f, indent=4)
```

---

## ⚠️ What Works vs What Doesn't

### ✅ Works Well

| Motion | Notes |
|--------|-------|
| Forward walk | Most reliable, model trained for this |
| Gentle turns | Combine with forward for smooth curves |
| Slow orbits | Quarter circles work better than full 360° |
| Look up/down | Gentle pitch changes |

### ⚠️ Can Be Problematic

| Motion | Issue |
|--------|-------|
| Fast orbits | May cause erratic results |
| Full 360° in 3 sec | Too fast, use longer duration |
| Complex combined motions | Model may get confused |
| Extreme rotations | Keep angles reasonable per frame |

### ❌ Doesn't Work

| Motion | Reason |
|--------|--------|
| Teleportation | Model expects continuous motion |
| Zoom (changing focal length) | Intrinsics should stay constant |
| Flying/swooping | Complex 3D paths confuse the model |

---

## 🔧 Recommended Parameters

```python
# Movement speed (per frame)
forward_speed = 0.08        # Units per frame (forward walk)
yaw_speed = np.deg2rad(3)   # 3° per frame (turning)
pitch_speed = np.deg2rad(2) # 2° per frame (looking up/down)
orbit_speed = np.deg2rad(5) # 5° per frame for orbits

# Orbit parameters
orbit_radius = 1.5 - 2.5    # Distance from subject
orbit_height = 0.2 - 0.5    # Height above subject

# Safe angle ranges per video
quarter_orbit = 90°         # For 3-5 second videos
half_orbit = 180°           # For 5+ second videos
full_orbit = 360°           # Needs 8+ seconds, may be unstable
```

---

## 📝 Quick Reference

```bash
# Generate with custom trajectory
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python generate.py \
  --image_path "input.png" \
  --prompt "Your scene description" \
  --pose_json_path "./assets/pose/my_trajectory.json" \
  --video_length 125 \
  --resolution 480p \
  --model_path ~/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/... \
  --action_ckpt ~/.cache/huggingface/hub/models--tencent--HY-WorldPlay/blobs/... \
  --model_type ar \
  --few_step true \
  --num_inference_steps 4 \
  --sr false \
  --output_path ./outputs/my_video.mp4
```

---

## 📚 Pre-made Trajectories

| File | Description | Duration |
|------|-------------|----------|
| `forward_20_latents.json` | Walk forward | 3.2 sec |
| `quarter_orbit_32_latents.json` | 90° orbit (behind → left) | 5.2 sec |
| `smooth_orbit_20_latents.json` | Gentle orbit + forward | 3.2 sec |

Located in `./assets/pose/`

---

## 🎓 Key Learnings

1. **WorldPlay simulates worlds, not just cameras** - Objects in the scene can move based on prompt and context.

2. **Prompt matters** - Describing camera movement in the prompt helps the model understand intent.

3. **Slower is better** - Gentle camera movements produce more stable results.

4. **Quarter orbits work best** - 90° orbits are more reliable than full 360°.

5. **Match duration to motion** - Complex motions need longer videos.

6. **Keep the subject centered** - For orbits, ensure camera always looks at the subject.
