import os
import gdown
from pathlib import Path

def download_model():
    model_path = Path("inswapper_128.onnx")
    if not model_path.exists():
        print("Downloading face swap model...")
        url = "https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF"
        gdown.download(url, str(model_path), quiet=False)
        print("Model downloaded successfully!")
    else:
        print("Model file already exists!")

if __name__ == "__main__":
    download_model() 