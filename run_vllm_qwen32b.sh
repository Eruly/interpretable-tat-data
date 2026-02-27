#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-27B on physical GPU 1,2,3,4
export CUDA_VISIBLE_DEVICES=1,2,3,4
# Keep compile path for speed, but reduce unstable Triton autotune behavior.
export TORCHINDUCTOR_MAX_AUTOTUNE=0

uv run --env-file .env vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --port 8000 \
  --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype auto \
  --served-model-name qwen3.5-27b
