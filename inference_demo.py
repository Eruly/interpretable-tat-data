import os
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Set HF_HOME to current directory
os.environ["HF_HOME"] = os.path.abspath(".")

def run_inference():
    # Load sample info
    if not os.path.exists("sample_info.json"):
        print("Error: sample_info.json not found. Run prepare_data.py first.")
        return

    with open("sample_info.json", "r") as f:
        sample_info = json.load(f)

    # Use a smaller model for demo verification
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print(f"Loading demo model {model_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Same prompt as planned for 235B
    prompt = (
        f"Question: {sample_info['question']}\n\n"
        "Analyze the table image and provide your reasoning. "
        "CRITICAL: For every numerical value or label you mention from the table, you MUST immediately follow it with its bounding box in the exact format: <bbox>(value, [ymin, xmin, ymax, xmax])<bbox>. "
        "The box coordinates must be normalized [0.0, 1.0].\n"
        "Example of reasoning: 'The revenue in 2019 was 218,096 <bbox>(218,096, [0.1, 0.2, 0.15, 0.3])<bbox>, which is higher than 2018.'\n"
        "Now, please answer the question with detailed reasoning."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "table.png"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generation
    print("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n--- Demo Response ---")
    print(output_text)

    # Save output
    with open("demo_output.txt", "w") as f:
        f.write(output_text)
    print("\nDemo response saved to demo_output.txt")

if __name__ == "__main__":
    run_inference()
