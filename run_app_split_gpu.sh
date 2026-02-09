#!/usr/bin/env bash
set -euo pipefail

# App process sees GPU0 for OCR and calls vLLM API for Qwen.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export OCR_DEVICE=cuda:0
export QWEN_BACKEND=vllm_api
export QWEN_API_URL=http://localhost:8000/v1
export QWEN_API_MODEL=qwen3-32b-thinking

uv run --env-file .env python app.py
