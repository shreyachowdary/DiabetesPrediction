"""
Full training pipeline: preprocess -> GA feature selection -> train -> save.

Run from project root:
    python scripts/run_training.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ml.preprocess import preprocess_pipeline
from ml.feature_selection_ga import run_ga_feature_selection
from ml.train import train_and_compare
from ml.evaluate import (
    compute_metrics,
    get_confusion_matrix,
    get_roc_curve_data,
    save_confusion_matrix_script,
    save_roc_curve_script,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    data_path = project_root / "data" / "diabetes.csv"
    models_dir = project_root / "models"
    scripts_dir = project_root / "scripts"
    models_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        logger.error("Dataset not found. Run: python scripts/download_data.py")
        sys.exit(1)
    
    # 1. Preprocess
    logger.info("Step 1: Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(
        data_path, test_size=0.2, random_state=42
    )
    
    # 2. Baseline (all features)
    logger.info("Step 2: Baseline model (all features)...")
    best_baseline, baseline_results, baseline_metrics = train_and_compare(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    print("\n--- Baseline (All Features) ---")
    for name, res in baseline_results.items():
        m = res["metrics"]
        print(f"  {name}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}, AUC={m['roc_auc']:.4f}")
    
    # 3. GA feature selection
    logger.info("Step 3: GA feature selection...")
    selected_names, selected_indices, _ = run_ga_feature_selection(
        X_train, y_train, feature_names,
        population_size=25,
        generations=12,
    )
    
    X_train_ga = X_train[:, selected_indices]
    X_test_ga = X_test[:, selected_indices]
    
    # 4. Train with selected features
    logger.info("Step 4: Training with GA-selected features...")
    best_model, ga_results, ga_metrics = train_and_compare(
        X_train_ga, X_test_ga, y_train, y_test, selected_names
    )
    
    print("\n--- After GA Feature Selection ---")
    print(f"  Selected features: {selected_names}")
    for name, res in ga_results.items():
        m = res["metrics"]
        print(f"  {name}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}, AUC={m['roc_auc']:.4f}")
    
    # 5. Save artifacts
    logger.info("Step 5: Saving artifacts...")
    
    joblib.dump(best_model, models_dir / "best_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(selected_names, models_dir / "selected_features.pkl")
    joblib.dump(feature_names, models_dir / "all_features.pkl")
    
    # Save metrics and model info for API
    model_info = {
        "best_model": max(ga_results.keys(), key=lambda k: ga_results[k]["metrics"]["f1_score"]),
        "selected_features": selected_names,
        "all_features": feature_names,
        "baseline_metrics": baseline_metrics,
        "ga_metrics": ga_metrics,
        "baseline_comparison": {
            name: res["metrics"] for name, res in baseline_results.items()
        },
        "ga_comparison": {
            name: res["metrics"] for name, res in ga_results.items()
        },
    }
    with open(models_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Confusion matrix and ROC scripts
    y_pred = best_model.predict(X_test_ga)
    y_prob = best_model.predict_proba(X_test_ga)
    cm = get_confusion_matrix(y_test, y_pred)
    roc_data = get_roc_curve_data(y_test, y_prob)
    
    save_confusion_matrix_script(scripts_dir / "plot_confusion_matrix.py", cm)
    save_roc_curve_script(
        scripts_dir / "plot_roc_curve.py",
        roc_data["fpr"],
        roc_data["tpr"],
        ga_metrics["roc_auc"],
    )
    
    # Feature importance for RF
    if "Random Forest" in ga_results and ga_results["Random Forest"]["feature_importance"]:
        imp = ga_results["Random Forest"]["feature_importance"]
        imp_script = scripts_dir / "plot_feature_importance.py"
        imp_script.write_text(f'''"""
Plot Random Forest feature importance.
Run: python scripts/plot_feature_importance.py
"""
import matplotlib.pyplot as plt

importance = {imp}
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
''')
    
    print("\n--- Training Complete ---")
    print(f"Best model: {model_info['best_model']}")
    print(f"Selected features: {selected_names}")
    print(f"Final metrics: {ga_metrics}")
    print(f"Artifacts saved to {models_dir}")


if __name__ == "__main__":
    main()
