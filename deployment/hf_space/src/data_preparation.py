"""Google Speech Commands v0.02 키워드 인식 데이터 준비 코드."""

import os
import numpy as np
import librosa
import tarfile
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import random

# 스마트홈 및 음성 비서 스타일 명령어
TARGET_KEYWORDS = [
    "yes", "no", "on", "off", "stop", "go", "up", "down", "left", "right"
]
# unknown: 위 명령어에 없는 단어
# silence: 말이 없거나 배경 소음만 있는 경우
CLASSES = TARGET_KEYWORDS + ["unknown", "silence"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

SAMPLE_RATE = 16000  # 초당 16,000개 샘플
N_MELS = 40
DURATION = 1.0  # 초
N_FFT = 512
HOP_LENGTH = 160
SILENCE_RMS_THRESHOLD = 0.008

def download_speech_commands(dest="data/raw"):
    "구글 음성 데이터셋 다운로드 -> 로컬에 저장"
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    tar_path = dest_path / "speech_commands_v0.02.tar.gz"
    extract_path = dest_path / "speech_commands"
    
    if not tar_path.exists():
        print("Downloading Speech Commands v0.02 (about 2.3GB)...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    
    if not extract_path.exists():
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_path)
        print("Extraction complete.")
    
    return extract_path

def preprocess_audio(audio_array, sr=SAMPLE_RATE, crop_active=True):
    """모노 변환, 음성 구간 선택, 리샘플링, 로그 멜 스펙트로그램 변환."""
    audio_array = np.asarray(audio_array)
    if np.issubdtype(audio_array.dtype, np.integer):
        audio_array = audio_array.astype(np.float32) / np.iinfo(audio_array.dtype).max
    else:
        audio_array = audio_array.astype(np.float32)
        if audio_array.size and np.max(np.abs(audio_array)) > 1.5:
            audio_array /= 32768.0

    # 스테레오 입력은 모노로 병합
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    if audio_array.size == 0:
        audio_array = np.zeros(int(sr * DURATION), dtype=np.float32)

    # 입력 전체를 정규화하지 않는다. 
    # 정규화하면 작은 마이크 잡음도 큰 소리처럼 바뀌게 된다.
    # 실시간 입력에서는 에너지가 가장 큰 1초 구간을 선택해 앞부분만 잘리는 문제를 줄인다.
    if crop_active and len(audio_array) > int(sr * DURATION):
        win = max(1, int(sr * 0.02))
        frame_count = max(1, len(audio_array) // win)
        trimmed = audio_array[:frame_count * win].reshape(frame_count, win)
        rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
        smooth = np.convolve(rms, np.ones(5, dtype=np.float32) / 5, mode="same")
        frames = max(1, int(DURATION * sr / win))
        best_frame = max(0, int(np.argmax(np.convolve(
            smooth, np.ones(frames, dtype=np.float32), mode="valid"
        ))))
        start = best_frame * win
        audio_array = audio_array[start:start + int(sr * DURATION)]

    # 샘플링 주파수를 16kHz
    if sr != SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # 로그 멜 스펙트로그램으로 바꿉니다. ref=1.0으로 입력의 실제 음량 차이를 보존합니다.
    mel = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)

    # 입력 길이는 1초
    target_frames = int(DURATION * SAMPLE_RATE / HOP_LENGTH)
    if log_mel.shape[1] < target_frames:
        pad_width = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    else:
        log_mel = log_mel[:, :target_frames]
    
    return log_mel.astype(np.float32)

def load_and_prepare_data(max_samples_per_class=300, seed=42):
    """데이터를 불러와 전처리하고 클래스별 수를 맞춘 뒤 분할합니다."""
    print("Preparing Speech Commands dataset...")
    random.seed(seed)
    dataset_path = download_speech_commands()
    
    data = []
    labels = []
    class_counts = {c: 0 for c in CLASSES}
    
    # 타겟 키워드 음성 파일 모으기
    audio_files = []
    for class_name in TARGET_KEYWORDS:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            random.shuffle(files)
            audio_files.extend([(f, class_name) for f in files[:max_samples_per_class * 2]])
    
    # 지정된 명령어가 아닌 다른 폴더에서 unknown 데이터를 수집
    other_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name not in TARGET_KEYWORDS and d.name != "_background_noise_"]
    for d in other_dirs[:5]:
        files = list(d.glob("*.wav"))
        random.shuffle(files)
        audio_files.extend([(f, "unknown") for f in files[:max_samples_per_class // 2]])
    
    # 긴 배경 소음 파일에서 서로 다른 1초 구간을 뽑고 작은 음량으로 합성해 silence 데이터를 만든다
    noise_dir = dataset_path / "_background_noise_"
    if noise_dir.exists():
        noise_files = list(noise_dir.glob("*.wav"))
        loaded_noise = [
            librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]
            for path in noise_files
        ]
        random.seed(seed)
        for _ in range(max_samples_per_class):
            noise_audio = random.choice(loaded_noise)
            clip_size = int(SAMPLE_RATE * DURATION)
            if len(noise_audio) < clip_size:
                noise_audio = np.pad(noise_audio, (0, clip_size - len(noise_audio)))
            start = random.randint(0, max(0, len(noise_audio) - clip_size))
            clip = noise_audio[start:start + clip_size].copy()
            clip *= random.uniform(0.02, 0.25)
            features = preprocess_audio(clip, SAMPLE_RATE, crop_active=False)
            data.append(features)
            labels.append(CLASS_TO_IDX["silence"])
            class_counts["silence"] += 1
    
    random.shuffle(audio_files)
    
    print(f"Found {len(audio_files)} candidate files. Processing...")
    
    for file_path, class_name in tqdm(audio_files, desc="Processing audio"):
        if class_counts[class_name] >= max_samples_per_class:
            continue
        
        try:
            audio, sr = librosa.load(file_path, sr=None)
            features = preprocess_audio(audio, sr)
            data.append(features)
            labels.append(CLASS_TO_IDX[class_name])
            class_counts[class_name] += 1
        except Exception as e:
            continue
        
        if all(v >= max_samples_per_class for v in class_counts.values()):
            break
    
    print("클래스별 데이터 수:", class_counts)
    
    X = np.array(data)
    y = np.array(labels)
    
    # 학습/검증/테스트 데이터 비율은 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    # 전처리된 데이터를 저장
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
