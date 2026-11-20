"""멜 스펙트로그램 기반 키워드 인식 기준 CNN 모델."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

try:
    from src.plot_style import configure_korean_font
except ImportError:
    from plot_style import configure_korean_font

configure_korean_font()

ROOT = Path(__file__).resolve().parents[1]

class MelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # 채널 차원을 추가합니다: (N, 1, 멜, 시간)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=12, n_mels=40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_baseline(epochs=20, batch_size=32, lr=0.001, device="cpu"):
    # 전처리된 데이터를 불러옵니다.
    with open(ROOT / "data" / "processed" / "data.pkl", "rb") as f:
        data = pickle.load(f)
    
    train_ds = MelDataset(data["X_train"], data["y_train"])
    val_ds = MelDataset(data["X_val"], data["y_val"])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    num_classes = len(data["classes"])
    model = BaselineCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증 데이터로 성능 계산
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, Val Loss={history['val_loss'][-1]:.4f}, Val Acc={val_acc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            (ROOT / "models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ROOT / "models" / "baseline_cnn.pth")
    
    # 학습 곡선 저장
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    best_epoch = int(np.argmax(history["val_f1"]))
    epochs_axis = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs_axis, history["train_loss"], marker="o", label="학습 손실")
    axes[0].plot(epochs_axis, history["val_loss"], marker="o", label="검증 손실")
    axes[0].set_title("손실: 낮을수록 좋음")
    axes[0].set_xlabel("에폭 (전체 학습 데이터 순회 횟수)")
    axes[0].set_ylabel("교차 엔트로피 손실")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[0].text(
        0.02, 0.03,
        "학습 손실만 감소하고 검증 손실이 오르면 과적합 신호일 수 있음",
        transform=axes[0].transAxes, fontsize=8,
    )

    axes[1].plot(epochs_axis, history["val_acc"], marker="o", label="정확도")
    axes[1].plot(epochs_axis, history["val_f1"], marker="o", label="Macro-F1")
    axes[1].axvline(best_epoch + 1, color="gray", linestyle="--", alpha=0.7)
    axes[1].scatter([best_epoch + 1], [history["val_f1"][best_epoch]], color="red", zorder=3)
    axes[1].annotate(
        f"선택 에폭 {best_epoch + 1}\nMacro-F1 {history['val_f1'][best_epoch]:.3f}",
        (best_epoch + 1, history["val_f1"][best_epoch]),
        xytext=(8, -28), textcoords="offset points", fontsize=8,
    )
    axes[1].set_title("검증 성능: 높을수록 좋음")
    axes[1].set_xlabel("에폭 (전체 학습 데이터 순회 횟수)")
    axes[1].set_ylabel("점수 (0~1)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("기준 CNN 학습 과정 | 저장 모델은 검증 Macro-F1 최고 에폭", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "baseline_training_curve.png", dpi=160)
    plt.close()
    
    print(f"최고 검증 Macro-F1: {best_val_f1:.4f} (에폭 {best_epoch + 1})")
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")
    train_baseline(epochs=15, device=device)
