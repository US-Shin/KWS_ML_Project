"""테스트 데이터에서 두 모델을 평가하고 비교 결과와 혼동행렬 저장하기."""

import os
import sys
import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# 이 파일을 직접 실행할 때 프로젝트 최상위 경로에서 모듈을 불러옵니다. (에러 해결을 위해 AI로 추가한 코드)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
PROJECT_ROOT = Path(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.plot_style import configure_korean_font
except ImportError:
    from plot_style import configure_korean_font

configure_korean_font()

from src.baseline_model import BaselineCNN, MelDataset
from src.proposed_model import ProposedResCNN

def evaluate_model(model, X_test, y_test, device, model_name):
    model.eval()
    dataset = MelDataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    # 혼동행렬 저장
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 클래스 이름 불러오기
    with open(PROJECT_ROOT / "data" / "processed" / "data.pkl", "rb") as f:
        data = pickle.load(f)
    class_names = data["classes"]
    
    row_totals = cm.sum(axis=1, keepdims=True)
    recall_by_row = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    # 색은 실제 클래스별 정답 비율, 칸의 문자는 개수와 실제 클래스 내 비율입니다.
    annotations = np.empty(cm.shape, dtype=object)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            annotations[row, col] = f"{cm[row, col]}\n{recall_by_row[row, col]:.0%}"

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        recall_by_row, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "실제 클래스 내 비율 (행 기준)"}, ax=ax,
    )
    ax.set_title(f"{model_name} 혼동행렬\n칸: 예측 개수 / 실제 클래스 내 비율 · 대각선이 정답")
    ax.set_ylabel("실제 클래스")
    ax.set_xlabel("모델 예측 클래스")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    safe_name = "_".join(model_name.lower().split())
    output_path = results_dir / f"{safe_name}_confusion_matrix.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    
    return acc, f1, classification_report(all_labels, all_preds)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(PROJECT_ROOT / "data" / "processed" / "data.pkl", "rb") as f:
        data = pickle.load(f)
    
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]
    num_classes = len(classes)
    
    # 베이스라인모델 로드
    baseline = BaselineCNN(num_classes=num_classes).to(device)
    baseline.load_state_dict(torch.load(PROJECT_ROOT / "models" / "baseline_cnn.pth", map_location=device))
    
    # Proposed모델 로드
    proposed = ProposedResCNN(num_classes=num_classes).to(device)
    proposed.load_state_dict(torch.load(PROJECT_ROOT / "models" / "proposed_rescnn.pth", map_location=device))
    
    print("=== Baseline Evaluation ===")
    b_acc, b_f1, b_report = evaluate_model(baseline, X_test, y_test, device, "Baseline CNN")
    print(f"Accuracy: {b_acc:.4f}, Macro F1: {b_f1:.4f}")
    print(b_report)
    
    print("\n=== Proposed Evaluation ===")
    p_acc, p_f1, p_report = evaluate_model(proposed, X_test, y_test, device, "Proposed ResCNN")
    print(f"Accuracy: {p_acc:.4f}, Macro F1: {p_f1:.4f}")
    print(p_report)
    
    # 모델 비교표를 저장
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame({
        "Model": ["Baseline CNN", "Proposed ResCNN"],
        "Accuracy": [b_acc, p_acc],
        "Macro F1": [b_f1, p_f1]
    })
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    # 두 지표를 통해 모델의 전체 성능과 클래스 균형을 함께 비교
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(comparison))
    width = 0.34
    bars_acc = ax.bar(positions - width / 2, comparison["Accuracy"], width, label="정확도")
    bars_f1 = ax.bar(positions + width / 2, comparison["Macro F1"], width, label="Macro-F1")
    ax.bar_label(bars_acc, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars_f1, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(positions, comparison["Model"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("점수 (0~1, 높을수록 좋음)")
    ax.set_title("테스트 성능 비교\nMacro-F1은 클래스별 F1을 동일한 비중으로 평균")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    guide = """# 시각화 읽는 법

## 학습 곡선
- 에폭은 학습 데이터 전체를 한 번 학습한 횟수입니다.
- 손실은 낮을수록 좋습니다. 학습 손실은 계속 내려가는데 검증 손실이 오르기 시작하면 과적합 가능성이 있습니다.
- 정확도는 전체 테스트/검증 샘플 중 맞힌 비율입니다.
- Macro-F1은 각 클래스의 F1을 같은 비중으로 평균합니다. 클래스별 데이터 수가 다를 때 정확도와 함께 봅니다.
- 점선/표식은 검증 Macro-F1이 가장 높아 체크포인트로 선택된 에폭입니다.

## 혼동행렬
- 세로 행은 실제 클래스, 가로 열은 모델 예측 클래스입니다.
- 각 칸은 `예측 개수`와 `그 실제 클래스 안에서 차지하는 비율`을 함께 표시합니다.
- 대각선은 정답입니다. 각 행의 대각선 비율은 해당 클래스 재현율이며, 대각선 밖 값은 잘못 분류된 사례입니다.
- 한 행의 테스트 샘플이 적으면 비율이 크게 흔들립니다. 행의 개수도 함께 확인하세요.

## 모델 비교 막대그래프
- 두 막대는 테스트 정확도와 Macro-F1입니다. 점수가 높을수록 좋습니다.
- 데이터 분할과 평가 조건이 같을 때 비교해야 하며, 점수만으로 실제 마이크 환경 성능이 보장되지는 않습니다.
"""
    (results_dir / "시각화_읽는법.md").write_text(guide, encoding="utf-8")
    print("\n=== Model Comparison ===")
    print(comparison)
    
    print("\nResults saved to results/ directory.")

if __name__ == "__main__":
    main()
