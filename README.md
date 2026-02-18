# Interpretable TAT Data

This repository contains the dataset and inference code for interpretable Table Question Answering (TAT-QA) using Qwen3-VL.

## Description

The project focuses on enhancing the interpretability of Vision-Language Models (VLMs) in table-related tasks. It uses a three-stage pipeline:
1. **Initial Draft Reasoning**: Qwen3-VL provides a draft answer.
2. **OCR Integration**: PaddleOCR is used to parse the table and provide ground-truth text and bounding boxes.
3. **Grounded Reasoning**: The model provides a final answer by referencing the OCR-parsed results and including bounding box coordinates in its reasoning steps.

## Directory Structure

- `data/`: Contains the processed TAT-QA samples and metadata.
- `app.py`: Gradio-based demo application.
- `inference_cls.py`: Core inference logic supporting vLLM and Transformers backends.
- `data_utils.py`: Utilities for data loading and table rendering.
- `modules/`: Helper modules for OCR and visualization.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   # Ensure transformers is installed from source for Qwen3-VL support
   pip install git+https://github.com/huggingface/transformers
   ```
2. Run the demo:
   ```bash
   python app.py
   ```

## License

This project is licensed under the Apache 2.0 License.