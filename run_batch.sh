#!/bin/bash
# HY-WorldPlay - Batch Video Generation (non-interactive)
# ========================================================
# Generates a video file from a prompt and camera trajectory JSON

cd "$(dirname "$0")"
source ~/anaconda3/bin/activate worldplay312

# Optional: prompt rewriting (requires vLLM server)
# export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
# export T2V_REWRITE_MODEL_NAME="<your_model_name>"
# export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
# export I2V_REWRITE_MODEL_NAME="<your_model_name>"

# Configuration
PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere.'

IMAGE_PATH=./assets/img/test.png
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p
OUTPUT_PATH=./outputs/
MODEL_PATH=/home/faraz/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/21d6fbc69d2eee5e6f932a8ab2068bb9f28c6fc7
AR_DISTILL_ACTION_MODEL_PATH=/home/faraz/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_distilled_action_model/model.safetensors
POSE_JSON_PATH=./assets/pose/test_forward_32_latents.json
NUM_FRAMES=13
WIDTH=832
HEIGHT=480
N_INFERENCE_GPU=1
REWRITE=false
ENABLE_SR=false

# Run batch generation (autoregressive distilled model, fast 4-step)
torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose_json_path $POSE_JSON_PATH \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
  --few_step true \
  --num_inference_steps 4 \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'ar'
