"""
Plot Random Forest feature importance.
Run: python scripts/plot_feature_importance.py
"""
import matplotlib.pyplot as plt

importance = {'Glucose': 0.40356949109537454, 'SkinThickness': 0.11698095538944682, 'BMI': 0.256521516204313, 'DiabetesPedigreeFunction': 0.22292803731086572}
names = list(importance.keys())
values = list(importance.values())
plt.figure(figsize=(8, 5))
plt.barh(names, values)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")
plt.close()
