# Interpretable TAT-QA

Interpretable Table Question Answering using Qwen3-VL with visual grounding.

## Pipeline

3-stage inference pipeline for explainable table QA:

1. **Stage 1 – Reasoning**: Qwen3-VL generates a draft reasoning answer from the table image.
2. **Stage 2 – OCR**: PaddleOCR extracts text and bounding boxes from the table.
3. **Stage 3 – Grounded Injection**: Qwen re-generates the reasoning with `<bbox>` coordinates injected, linking each referenced value to its position on the table.

## Project Structure

```
.
├── app.py                  # FastAPI server (API endpoints + static file serving)
├── inference_cls.py        # 3-stage inference engine (vLLM API / Transformers)
├── data_utils.py           # Data loading & table-to-PNG rendering
├── static/                 # Frontend (vanilla HTML/JS/CSS)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── data/                   # Dataset
│   ├── train.json
│   └── sample_info.json
├── scripts/                # Launch & utility scripts
│   ├── run_app_split_gpu.sh
│   ├── run_vllm_qwen32b.sh
│   ├── run_vllm_docker.sh  # PaddleOCR Docker
│   ├── download_model.py
│   ├── prepare_data.py
│   ├── inference.py         # Standalone inference script
│   └── inference_demo.py
├── tests/                  # Test scripts
│   ├── test_bbox_parsing.py
│   ├── test_ocr_*.py
│   └── test_load_32b.py
├── chat_template.json
└── .gitignore
```

## Quick Start

### 1. Start vLLM (Qwen3.5-27B)

```bash
bash scripts/run_vllm_qwen32b.sh
```

### 2. Start PaddleOCR Server

```bash
bash scripts/run_vllm_docker.sh
```

### 3. Run the Web App

```bash
bash scripts/run_app_split_gpu.sh
```

Open `http://localhost:7862` in your browser.

## Configuration

Environment variables are set in `scripts/run_app_split_gpu.sh`:

| Variable | Description | Default |
|----------|-------------|---------|
| `QWEN_BACKEND` | `vllm_api` or `transformers` | `vllm_api` |
| `QWEN_API_URL` | vLLM endpoint URL | `http://localhost:8000/v1` |
| `QWEN_API_MODEL` | Model name on vLLM | `qwen3.5-27b` |
| `OCR_SERVER_URL` | PaddleOCR vLLM endpoint | `http://localhost:8080/v1` |
| `temperature` | Sampling temperature | `0.6` |

## License

Apache 2.0
