#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-32B-Thinking on physical GPU 1,2,3,4
export CUDA_VISIBLE_DEVICES=1,2,3,4

uv run --env-file .env vllm serve hub/models--Qwen--Qwen3-VL-32B-Thinking \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --port 8000 \
  --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype auto \
  --served-model-name qwen3-32b-thinking
