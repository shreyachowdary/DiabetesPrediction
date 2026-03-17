"""
Model training pipeline.

Trains Random Forest, SVM, and optionally Logistic Regression.
Performs hyperparameter tuning and saves the best model.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from ml.evaluate import compute_metrics, get_confusion_matrix, get_roc_curve_data

logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict] = None,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train Random Forest with optional GridSearch.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [6, 8, 10, None],
            "min_samples_split": [2, 5],
        }
    
    base_rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        base_rf,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model, {
        "best_params": grid_search.best_params_,
        "cv_score": float(grid_search.best_score_),
    }


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict] = None,
) -> Tuple[SVC, Dict[str, Any]]:
    """
    Train SVM with optional GridSearch.
    Uses probability=True for predict_proba support.
    """
    if param_grid is None:
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }
    
    base_svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(
        base_svm,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model, {
        "best_params": grid_search.best_params_,
        "cv_score": float(grid_search.best_score_),
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Train Logistic Regression as baseline.
    """
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train, y_train)
    return lr, {"best_params": {"C": 1.0}}


def get_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract feature importance from Random Forest.
    Returns empty dict for non-tree models.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return dict(zip(feature_names, imp.tolist()))
    return {}


def train_and_compare(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Train RF, SVM, LR and return the best model with metrics.
    """
    results = {}
    
    # Random Forest
    rf_model, rf_info = train_random_forest(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)
    results["Random Forest"] = {
        "model": rf_model,
        "metrics": compute_metrics(y_test, y_pred_rf, y_prob_rf),
        "feature_importance": get_feature_importance(rf_model, feature_names),
        "info": rf_info,
    }
    
    # SVM
    svm_model, svm_info = train_svm(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    y_prob_svm = svm_model.predict_proba(X_test)
    results["SVM"] = {
        "model": svm_model,
        "metrics": compute_metrics(y_test, y_pred_svm, y_prob_svm),
        "feature_importance": {},
        "info": svm_info,
    }
    
    # Logistic Regression (baseline)
    lr_model, lr_info = train_logistic_regression(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)
    results["Logistic Regression"] = {
        "model": lr_model,
        "metrics": compute_metrics(y_test, y_pred_lr, y_prob_lr),
        "feature_importance": {},
        "info": lr_info,
    }
    
    # Select best by F1 (good balance of precision/recall for medical use)
    best_name = max(
        results.keys(),
        key=lambda k: results[k]["metrics"]["f1_score"],
    )
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]
    
    logger.info(f"Best model: {best_name} (F1={best_metrics['f1_score']:.4f})")
    
    return best_model, results, best_metrics
