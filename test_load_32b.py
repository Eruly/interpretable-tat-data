import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os

model_path = "/home/csle/DATA/dkyoon/interpretable-llm/table_interpretable/hub/models--Qwen--Qwen3-VL-32B-Thinking"

print(f"Loading model from {model_path}...")
try:
    # Use device_map="auto" to distribute across GPUs if needed, or just one GPU for test
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
