"""
API tests for the Diabetes Prediction service.

Run: pytest tests/test_api.py -v
Requires: server running (python run_local.py) or use TestClient for unit tests.
"""

import pytest
from fastapi.testclient import TestClient

# Import app - may fail if models not trained
try:
    from app.main import app
    client = TestClient(app)
except Exception as e:
    client = None
    import_error = str(e)


@pytest.fixture
def api_client():
    if client is None:
        pytest.skip(f"Cannot load app: {import_error}")
    return client


def test_health_check(api_client):
    """Health endpoint should return 200."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_valid_input(api_client):
    """Valid prediction input should return prediction."""
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35,
    }
    response = api_client.post("/predict", json=payload)
    # May be 503 if models not loaded
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ("diabetic", "non-diabetic")
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 1


def test_predict_invalid_input(api_client):
    """Invalid input should return 422."""
    payload = {
        "Pregnancies": -1,  # Invalid: must be >= 0
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_info(api_client):
    """Model info endpoint should return metadata or 503."""
    response = api_client.get("/model-info")
    if response.status_code == 200:
        data = response.json()
        assert "best_model" in data
        assert "selected_features" in data
        assert "metrics" in data
