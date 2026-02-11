"""
Evaluate the trained ANN model on the held-out test set.
- Confusion matrix, classification report, ROC-AUC, PR-AUC
- Saves plots as PNGs to the models/ directory
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from tensorflow import keras

from src.config import DATA_PROCESSED, MODEL_PATH, MODEL_DIR


def load_test_data():
    X_test = np.load(DATA_PROCESSED / "X_test.npy")
    y_test = np.load(DATA_PROCESSED / "y_test.npy")
    print(f"✔ Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    return X_test, y_test


def evaluate():
    X_test, y_test = load_test_data()

    # Load model
    model = keras.models.load_model(MODEL_PATH)
    print(f"✔ Model loaded from {MODEL_PATH}")

    # Predict probabilities
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    # ── Confusion Matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"✔ Saved → {cm_path}")

    # ── ROC Curve ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = MODEL_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"✔ Saved → {roc_path}")

    # ── Precision-Recall Curve ───────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    pr_path = MODEL_DIR / "pr_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"✔ Saved → {pr_path}")

    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    evaluate()
