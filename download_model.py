"""
Download TRIBE v2 model weights from HuggingFace.
Run this once on a new deployment before starting the app.

Usage:
    python download_model.py
"""
import os

def download():
    from huggingface_hub import snapshot_download
    
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Downloading TRIBE v2 model weights...")
    snapshot_download(
        repo_id="pbhatt17/TRIBE-2",
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to {model_dir}")

if __name__ == "__main__":
    download()
