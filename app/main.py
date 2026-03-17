"""
FastAPI application for diabetes prediction.

Serves prediction endpoints and static frontend.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.model_loader import get_artifacts
from app.schemas import PredictionInput, PredictionResponse, ModelInfoResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetes Prediction API",
    description="ML-based diabetes prediction using GA feature selection",
    version="1.0.0",
)

# Resolve paths for static files
def _get_static_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "static"


def _get_index_path() -> Path:
    return _get_static_path() / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = _get_index_path()
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>Diabetes Prediction API</h1><p>Frontend not found. Run from project root.</p>"
    )


@app.get("/health")
async def health_check():
    """Health check for deployment monitoring."""
    return {"status": "ok", "service": "diabetes-prediction"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """
    Predict diabetes risk from health metrics.
    Returns prediction label and confidence.
    """
    try:
        artifacts = get_artifacts()
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        selected_features = artifacts["selected_features"]
        all_features = artifacts["all_features"]
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    # Build input array in correct order (all features for scaler)
    input_dict = input_data.model_dump()
    X_full = np.array([[input_dict[f] for f in all_features]])
    X_scaled_full = scaler.transform(X_full)
    # Select GA-chosen features for the model
    selected_indices = [all_features.index(f) for f in selected_features]
    X_scaled = X_scaled_full[:, selected_indices]

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    # Probability of positive class (diabetic)
    prob_diabetic = float(proba[1]) if len(proba) > 1 else float(proba[0])
    label = "diabetic" if prediction == 1 else "non-diabetic"
    confidence = prob_diabetic if prediction == 1 else (1 - prob_diabetic)

    logger.info(f"Prediction: {label} (confidence={confidence:.2f})")

    return PredictionResponse(
        prediction=label,
        probability=prob_diabetic,
        confidence=round(confidence, 4),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata, selected features, and metrics."""
    try:
        artifacts = get_artifacts()
        info = artifacts.get("model_info", {})
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    return ModelInfoResponse(
        best_model=info.get("best_model", "Unknown"),
        selected_features=info.get("selected_features", []),
        metrics=info.get("ga_metrics", {}),
        baseline_comparison=info.get("baseline_comparison", {}),
        ga_comparison=info.get("ga_comparison", {}),
    )


# Mount static files for CSS/JS (only if directory exists)
static_path = _get_static_path()
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
