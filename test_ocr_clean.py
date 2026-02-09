import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def render_clean_table(table_data, output_path="clean_table.png"):
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.7 + 1.5))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.2)
    plt.savefig(output_path, dpi=200, transparent=False, facecolor='white')
    plt.close()
    return output_path

def test_ocr_clean():
    model_id = "lightonai/LightOnOCR-2-1B"
    print(f"Loading {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Render clean image
    with open("train.json", "r") as f:
        samples = json.load(f)
    image_path = render_clean_table(samples[0]["table"]["table"], "clean_table.png")
    image = Image.open(image_path).convert("RGB")
    
    conversation = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
    
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=4096)
    
    output_text = processor.decode(output_ids[0], skip_special_tokens=True)
    print("--- CLEAN OCR OUTPUT ---")
    print(output_text)
    print("------------------------")

if __name__ == "__main__":
    test_ocr_clean()
