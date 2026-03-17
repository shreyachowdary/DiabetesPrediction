"""
Generated script to plot confusion matrix.
Run: python plot_confusion_matrix.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array([[84, 16], [27, 27]])
labels = ["Non-Diabetic", "Diabetic"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
plt.close()
