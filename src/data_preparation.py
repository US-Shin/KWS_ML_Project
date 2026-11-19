"""
Data Preparation for Keyword Spotting Project
Dataset: Google Speech Commands v0.02 (via pollen-robotics/speech-commands-v0.02)
"""

import os
import numpy as np
import librosa
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

# Selected keywords for smart home / voice assistant style commands
TARGET_KEYWORDS = [
    "yes", "no", "on", "off", "stop", "go", "up", "down", "left", "right"
]
# We will also have "unknown" and "silence" classes
CLASSES = TARGET_KEYWORDS + ["unknown", "silence"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

SAMPLE_RATE = 16000
N_MELS = 40
DURATION = 1.0  # seconds
N_FFT = 512
HOP_LENGTH = 160

def preprocess_audio(audio_array, sr=SAMPLE_RATE):
    """Preprocessing steps:
    1. Resample to 16kHz (if needed)
    2. Normalize amplitude
    3. Extract Log-Mel Spectrogram
    4. Pad or trim to fixed length
    """
    # Ensure mono and correct sample rate
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # Normalize
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    
    # Compute Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Pad or trim to fixed size (time frames for 1 second)
    target_frames = int(DURATION * sr / HOP_LENGTH)
    if log_mel.shape[1] < target_frames:
        pad_width = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :target_frames]
    
    return log_mel.astype(np.float32)

def load_and_prepare_data(max_samples_per_class=500, seed=42):
    """Load dataset, filter classes, preprocess, and split."""
    print("Loading dataset from Hugging Face...")
    # Using the parquet version for compatibility
    dataset = load_dataset("pollen-robotics/speech-commands-v0.02", split="train")
    
    # Filter and sample
    data = []
    labels = []
    
    class_counts = {c: 0 for c in CLASSES}
    
    for example in tqdm(dataset, desc="Filtering and sampling"):
        label = example["label"]  # assuming the column is 'label' or adjust based on actual
        # Note: Check actual column names. Common is 'audio' and 'label' or 'transcription'
        # For pollen-robotics version, typically 'audio' and 'label'
        
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        
        # Map label
        if label in TARGET_KEYWORDS:
            class_name = label
        elif label in ["_background_noise_", "silence"] or "silence" in str(label).lower():
            class_name = "silence"
        else:
            class_name = "unknown"
        
        if class_counts[class_name] >= max_samples_per_class:
            continue
        
        try:
            features = preprocess_audio(audio, sr)
            data.append(features)
            labels.append(CLASS_TO_IDX[class_name])
            class_counts[class_name] += 1
        except Exception as e:
            continue
        
        if all(v >= max_samples_per_class for v in class_counts.values()):
            break
    
    print("Class distribution:", class_counts)
    
    X = np.array(data)
    y = np.array(labels)
    
    # Train / Val / Test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/data.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "classes": CLASSES,
            "class_to_idx": CLASS_TO_IDX
        }, f)
    
    print(f"Saved processed data. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    load_and_prepare_data(max_samples_per_class=300)
