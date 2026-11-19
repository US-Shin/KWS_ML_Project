"""잔차 연결을 사용한 제안형 CNN 모델."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from src.plot_style import configure_korean_font
except ImportError:
    from plot_style import configure_korean_font

configure_korean_font()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ProposedResCNN(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

def train_proposed(epochs=25, batch_size=32, lr=0.0005, device="cpu"):
    with open(ROOT / "data" / "processed" / "data.pkl", "rb") as f:
        data = pickle.load(f)
    
    train_ds = MelDataset(data["X_train"], data["y_train"])
    val_ds = MelDataset(data["X_val"], data["y_val"])
    
    # 학습/검증용 데이터 묶음을 만듭니다.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    num_classes = len(data["classes"])
    model = ProposedResCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f"Proposed Epoch {epoch+1}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 검증 데이터로 성능 확인
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
        
        print(f"에폭 {epoch+1}: 검증 정확도={val_acc:.4f}, 검증 Macro-F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            (ROOT / "models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ROOT / "models" / "proposed_rescnn.pth")
    
    best_epoch = int(np.argmax(history["val_f1"]))
    epochs_axis = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(epochs_axis, history["train_loss"], marker="o", label="학습 손실")
    axes[0].plot(epochs_axis, history["val_loss"], marker="o", label="검증 손실")
    axes[0].set_title("손실: 낮을수록 좋음")
    axes[0].set_xlabel("에폭 (전체 학습 데이터 순회 횟수)")
    axes[0].set_ylabel("교차 엔트로피 손실")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].text(
        0.02, 0.03,
        "학습 손실만 감소하고 검증 손실이 오르면 과적합 신호일 수 있음",
        transform=axes[0].transAxes, fontsize=8,
    )

    axes[1].plot(epochs_axis, history["val_acc"], marker="o", color="tab:blue")
    axes[1].set_title("검증 정확도: 전체 정답 비율")
    axes[1].set_xlabel("에폭")
    axes[1].set_ylabel("점수 (0~1)")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs_axis, history["val_f1"], marker="o", color="tab:green")
    axes[2].axvline(best_epoch + 1, color="gray", linestyle="--", alpha=0.7)
    axes[2].scatter([best_epoch + 1], [history["val_f1"][best_epoch]], color="red", zorder=3)
    axes[2].annotate(
        f"저장 에폭 {best_epoch + 1}\nMacro-F1 {history['val_f1'][best_epoch]:.3f}",
        (best_epoch + 1, history["val_f1"][best_epoch]),
        xytext=(8, -28), textcoords="offset points", fontsize=8,
    )
    axes[2].set_title("검증 Macro-F1: 클래스별 성능 균형")
    axes[2].set_xlabel("에폭")
    axes[2].set_ylabel("점수 (0~1)")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.25)

    fig.suptitle("제안 ResCNN 학습 과정 | 점선은 검증 Macro-F1 최고 에폭", fontsize=12)
    fig.tight_layout()
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    plt.savefig(ROOT / "results" / "proposed_training_curve.png")
    plt.close()
    
    print(f"최고 검증 Macro-F1: {best_val_f1:.4f}")
    return model

# 데이터셋 클래스로 멜 스펙트로그램과 정답 라벨을 텐서로 변환
class MelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")
    train_proposed(epochs=20, device=device)
