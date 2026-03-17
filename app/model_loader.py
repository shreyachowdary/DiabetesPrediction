"""
Load trained model artifacts for inference.

Handles paths for both local development and Vercel deployment.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib

logger = logging.getLogger(__name__)

# Resolve project root: works for both local and Vercel
def _get_project_root() -> Path:
    """Get project root - works when running from api/ or project root."""
    current = Path(__file__).resolve()
    # app/model_loader.py -> project root
    if current.parent.name == "app":
        return current.parent.parent
    # api/index.py imports from app
    return current.parent


def load_model_artifacts() -> dict:
    """
    Load model, scaler, and selected features.
    Returns dict with keys: model, scaler, selected_features, all_features, model_info
    """
    root = _get_project_root()
    models_dir = root / "models"

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    artifacts = {}

    model_path = models_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found. Run training first: {model_path}")

    artifacts["model"] = joblib.load(model_path)
    artifacts["scaler"] = joblib.load(models_dir / "scaler.pkl")
    artifacts["selected_features"] = joblib.load(models_dir / "selected_features.pkl")
    artifacts["all_features"] = joblib.load(models_dir / "all_features.pkl")

    info_path = models_dir / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            artifacts["model_info"] = json.load(f)
    else:
        artifacts["model_info"] = {}

    logger.info("Model artifacts loaded successfully")
    return artifacts


# Module-level cache to avoid reloading on every request
_cache: Optional[dict] = None


def get_artifacts() -> dict:
    """Get cached artifacts or load fresh."""
    global _cache
    if _cache is None:
        _cache = load_model_artifacts()
    return _cache
