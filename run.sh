#!/bin/bash
# HY-WorldPlay - Video Generation Script
# ==========================================
# Usage: ./run.sh [MODEL]
#   MODEL: wan (default), hunyuan-bi, hunyuan-ar, hunyuan-distilled
#
# For interactive demo, use: ./run_interactive.sh

set -e

# Activate conda environment
source ~/anaconda3/bin/activate worldplay312

cd "$(dirname "$0")"

MODEL="${1:-wan}"

# ==========================================
# Configuration
# ==========================================
PROMPT='Dark lord walking toward camera'
IMAGE_PATH=/wrk/tmt/movie1/voice_design/game/king/courtyard.png
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p
OUTPUT_PATH=./outputs/
# Camera trajectory: w-N gives 1 + N*4 frames
# HunyuanVideo latents must be divisible by 4: w-7 -> 29 frames (8 latents), w-31 -> 125 frames (32 latents)
POSE='w-7'
NUM_FRAMES=29  # For w-7 pose (8 latents, divisible by 4)
WIDTH=832
HEIGHT=480

# HunyuanVideo model paths
HUNYUAN_SNAPSHOT="/home/faraz/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/21d6fbc69d2eee5e6f932a8ab2068bb9f28c6fc7"
WORLDPLAY_SNAPSHOT="/home/faraz/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e"

MODEL_PATH="$HUNYUAN_SNAPSHOT"
AR_ACTION_MODEL_PATH="$WORLDPLAY_SNAPSHOT/ar_model/diffusion_pytorch_model.safetensors"
BI_ACTION_MODEL_PATH="$WORLDPLAY_SNAPSHOT/bidirectional_model/diffusion_pytorch_model.safetensors"
AR_DISTILL_ACTION_MODEL_PATH="$WORLDPLAY_SNAPSHOT/ar_distilled_action_model/model.safetensors"

# WAN model path
WAN_SNAPSHOT="/home/faraz/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/969249711ab41203e8d8d9f784a82372e9070ac5/"

# Multi-GPU config for HunyuanVideo (auto-detect GPU count)
N_INFERENCE_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
N_INFERENCE_GPU=${N_INFERENCE_GPU:-1}
REWRITE=false
ENABLE_SR=false

# ==========================================
# Run selected model
# ==========================================
echo "Running with model: $MODEL"

case "$MODEL" in
  wan)
    echo "Using WAN model (faster inference)"
    PYTHONPATH=/wrk/tmt/HY-WorldPlay/wan:/wrk/tmt/HY-WorldPlay:$PYTHONPATH \
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 \
    python wan/generate.py \
      --input "$PROMPT" \
      --image_path "$IMAGE_PATH" \
      --num_chunk 2 \
      --pose "$POSE" \
      --ar_model_path "${WAN_SNAPSHOT}wan_transformer" \
      --ckpt_path "${WAN_SNAPSHOT}wan_distilled_model/model.pt" \
      --out "$OUTPUT_PATH"
    ;;

  hunyuan-bi)
    echo "Using HunyuanVideo bidirectional model"
    PYTHONPATH=/wrk/tmt/HY-WorldPlay:$PYTHONPATH \
    torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
      --prompt "$PROMPT" \
      --image_path "$IMAGE_PATH" \
      --resolution $RESOLUTION \
      --aspect_ratio $ASPECT_RATIO \
      --video_length $NUM_FRAMES \
      --seed $SEED \
      --rewrite $REWRITE \
      --sr $ENABLE_SR --save_pre_sr_video \
      --pose "$POSE" \
      --output_path $OUTPUT_PATH \
      --model_path $MODEL_PATH \
      --action_ckpt $BI_ACTION_MODEL_PATH \
      --few_step false \
      --model_type 'bi'
    ;;

  hunyuan-ar)
    echo "Using HunyuanVideo autoregressive model"
    PYTHONPATH=/wrk/tmt/HY-WorldPlay:$PYTHONPATH \
    torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
      --prompt "$PROMPT" \
      --image_path "$IMAGE_PATH" \
      --resolution $RESOLUTION \
      --aspect_ratio $ASPECT_RATIO \
      --video_length $NUM_FRAMES \
      --seed $SEED \
      --rewrite $REWRITE \
      --sr $ENABLE_SR --save_pre_sr_video \
      --pose "$POSE" \
      --output_path $OUTPUT_PATH \
      --model_path $MODEL_PATH \
      --action_ckpt $AR_ACTION_MODEL_PATH \
      --few_step false \
      --width $WIDTH \
      --height $HEIGHT \
      --model_type 'ar'
    ;;

  hunyuan-distilled)
    echo "Using HunyuanVideo autoregressive distilled model"
    PYTHONPATH=/wrk/tmt/HY-WorldPlay:$PYTHONPATH \
    torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
      --prompt "$PROMPT" \
      --image_path "$IMAGE_PATH" \
      --resolution $RESOLUTION \
      --aspect_ratio $ASPECT_RATIO \
      --video_length $NUM_FRAMES \
      --seed $SEED \
      --rewrite $REWRITE \
      --sr $ENABLE_SR --save_pre_sr_video \
      --pose "$POSE" \
      --output_path $OUTPUT_PATH \
      --model_path $MODEL_PATH \
      --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
      --few_step true \
      --num_inference_steps 4 \
      --model_type 'ar' \
      --use_vae_parallel false \
      --use_sageattn false \
      --use_fp8_gemm false
    ;;

  *)
    echo "Unknown model: $MODEL"
    echo "Available models: wan, hunyuan-bi, hunyuan-ar, hunyuan-distilled"
    exit 1
    ;;
esac

echo "Done!"
