from inference_cls import QwenInference
import json
import os

def run_inference():
    # Load sample info
    if not os.path.exists("sample_info.json"):
        print("Error: sample_info.json not found. Run prepare_data.py first.")
        return

    with open("sample_info.json", "r") as f:
        sample_info = json.load(f)

    infer = QwenInference(model_id="Qwen/Qwen3-VL-235B-A22B-Thinking")
    output_text = infer.get_answer("table.png", sample_info['question'])

    print("\n--- Response ---")
    print(output_text)

    # Save output
    with open("output.txt", "w") as f:
        f.write(output_text)
    print("\nResponse saved to output.txt")

if __name__ == "__main__":
    run_inference()
