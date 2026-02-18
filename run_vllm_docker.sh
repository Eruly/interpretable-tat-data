#!/usr/bin/env bash
set -euo pipefail

# Container config
IMAGE="vllm/vllm-openai:latest"
CONTAINER_NAME="vllm-qwen32b"
# Use specific GPUs 1,2,3,4
DEVICES='"device=1,2,3,4"'

# Model config
# We mount the local hub directory to the container's HF cache
MODEL_REPO="Qwen/Qwen3-VL-32B-Thinking"
# vLLM inside docker typically runs as root
HF_Home="/root/.cache/huggingface"

echo "Starting vLLM server in Docker..."
echo "Using GPUs: $DEVICES"
echo "Model: $MODEL_REPO"

# Remove existing container if it exists
docker rm -f $CONTAINER_NAME || true

# Run container
# --ipc=host is crucial for vLLM distributed inference
docker run -d \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --gpus $DEVICES \
    -v $(pwd)/hub:$HF_Home/hub \
    -p 8000:8000 \
    --ipc=host \
    $IMAGE \
    --model $MODEL_REPO \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.9 \
    --kv-cache-dtype auto \
    --served-model-name qwen3-32b-thinking \
    --trust-remote-code

echo "Container $CONTAINER_NAME started."
echo "Follow logs with: docker logs -f $CONTAINER_NAME"
