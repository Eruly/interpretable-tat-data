#!/usr/bin/env bash
set -euo pipefail

# Run from project root
cd "$(dirname "$0")/.."

# App process sees GPU0 for OCR and calls vLLM API for Qwen.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export OCR_DEVICE=cuda:0
export QWEN_BACKEND=vllm_api
# export QWEN_API_URL=http://localhost:8000/v1
# export QWEN_API_MODEL=qwen3.5-27b

export QWEN_API_URL=http://10.77.0.194:8000/v1
export QWEN_API_MODEL="Qwen/Qwen3.5-397B-A17B"
export OCR_SERVER_URL=http://localhost:8080/v1

export temperature=0.6
export top_p=0.95
export top_k=20
export min_p=0.0
export presence_penalty=0.0
export repetition_penalty=1.0



uv run --env-file .env uvicorn app:app --host 0.0.0.0 --port "${GRADIO_SERVER_PORT:-7862}"
