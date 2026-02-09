import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

model_id = "Qwen/Qwen3-VL-235B-A22B-Thinking"
target_dir = os.environ.get("HF_HOME", ".")

print(f"Starting download of {model_id} to {target_dir}...")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Download completed successfully!")
except Exception as e:
    print(f"Error during download: {e}")
