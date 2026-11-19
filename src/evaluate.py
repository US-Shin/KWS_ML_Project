"""
Evaluation script for both models on the test set.
Computes Accuracy, Macro F1, and generates comparison table + confusion matrices.
"""

import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(f"results/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    
    return acc, f1, classification_report(all_labels, all_preds)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open("data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)
    
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]
    num_classes = len(classes)
    
    # Load Baseline
    baseline = BaselineCNN(num_classes=num_classes).to(device)
    baseline.load_state_dict(torch.load("models/baseline_cnn.pth", map_location=device))
    
    # Load Proposed
    proposed = ProposedResCNN(num_classes=num_classes).to(device)
    proposed.load_state_dict(torch.load("models/proposed_rescnn.pth", map_location=device))
    
    print("=== Baseline Evaluation ===")
    b_acc, b_f1, b_report = evaluate_model(baseline, X_test, y_test, device, "Baseline CNN")
    print(f"Accuracy: {b_acc:.4f}, Macro F1: {b_f1:.4f}")
    print(b_report)
    
    print("\n=== Proposed Evaluation ===")
    p_acc, p_f1, p_report = evaluate_model(proposed, X_test, y_test, device, "Proposed ResCNN")
    print(f"Accuracy: {p_acc:.4f}, Macro F1: {p_f1:.4f}")
    print(p_report)
    
    # Comparison Table
    comparison = pd.DataFrame({
        "Model": ["Baseline CNN", "Proposed ResCNN"],
        "Accuracy": [b_acc, p_acc],
        "Macro F1": [b_f1, p_f1]
    })
    comparison.to_csv("results/model_comparison.csv", index=False)
    print("\n=== Model Comparison ===")
    print(comparison)
    
    print("\nResults saved to results/ directory.")

if __name__ == "__main__":
    main()
