import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import base64
import io

def test_ocr():
    model_id = "lightonai/LightOnOCR-2-1B-ocr-soup"
    print(f"Loading {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Note: LightOnOCR-2-1B might require specific class or just Auto
    # README says LightOnOcrForConditionalGeneration, checking if it's in the current transformers
    try:
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        model = LightOnOcrForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        processor = LightOnOcrProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Specific class failed: {e}. Trying Auto...")
        model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Use a test image (e.g., the one rendered before)
    image_path = "tmp_table_0.png"
    if not os.path.exists(image_path):
        # Create a dummy image if needed, or just skip
        print(f"Image {image_path} not found.")
        return

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
    # Move to device and ensure consistent dtype for floating point tensors
    inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
    
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=4096)
    
    # Trim input IDs as per HF README
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)
    
    print("--- RAW OCR OUTPUT ---")
    print(repr(output_text))
    print("--- DECODED OCR OUTPUT ---")
    print(output_text)
    print("------------------")

if __name__ == "__main__":
    import os
    test_ocr()
