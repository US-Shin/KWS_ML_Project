# Alternative: Download official Speech Commands and process with librosa
# Run this if Hugging Face loading has issues

import os
import tarfile
import urllib.request
from pathlib import Path

def download_speech_commands(dest="data/raw"):
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    tar_path = dest_path / "speech_commands_v0.02.tar.gz"
    
    if not tar_path.exists():
        print("Downloading Speech Commands v0.02...")
        urllib.request.urlretrieve(url, tar_path)
    
    extract_path = dest_path / "speech_commands"
    if not extract_path.exists():
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_path)
    print(f"Dataset ready at {extract_path}")
    return extract_path

if __name__ == "__main__":
    download_speech_commands()
