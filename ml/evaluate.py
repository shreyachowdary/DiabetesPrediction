"""
Model evaluation utilities.

Computes accuracy, precision, recall, F1-score, ROC-AUC,
and generates confusion matrix and ROC curve data.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    ROC-AUC only if y_prob is provided and problem is binary.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            # Use probability of positive class
            if y_prob.ndim > 1:
                y_prob_pos = y_prob[:, 1]
            else:
                y_prob_pos = y_prob
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
    
    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return confusion matrix as 2x2 array."""
    return confusion_matrix(y_true, y_pred)


def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, list]:
    """
    Get FPR, TPR, thresholds for ROC curve plotting.
    """
    if y_prob.ndim > 1:
        y_prob_pos = y_prob[:, 1]
    else:
        y_prob_pos = y_prob
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def save_confusion_matrix_script(
    output_path: Path,
    cm: np.ndarray,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generate a standalone script to plot and save confusion matrix.
    """
    script = f'''"""
Generated script to plot confusion matrix.
Run: python {output_path.name}
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array({cm.tolist()})
labels = ["Non-Diabetic", "Diabetic"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("{title}")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
plt.close()
'''
    output_path.write_text(script)
    logger.info(f"Saved confusion matrix script to {output_path}")


def save_roc_curve_script(
    output_path: Path,
    fpr: list,
    tpr: list,
    roc_auc: float,
) -> None:
    """
    Generate a standalone script to plot and save ROC curve.
    """
    script = f'''"""
Generated script to plot ROC curve.
Run: python {output_path.name}
"""
import matplotlib.pyplot as plt
import numpy as np

fpr = {fpr}
tpr = {tpr}
roc_auc = {roc_auc}

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {{roc_auc:.3f}})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
print("Saved roc_curve.png")
plt.close()
'''
    output_path.write_text(script)
    logger.info(f"Saved ROC curve script to {output_path}")
