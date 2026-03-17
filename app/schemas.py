"""
Pydantic models for request/response validation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input schema for diabetes prediction."""

    Pregnancies: int = Field(ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(ge=0, le=300, description="Plasma glucose concentration")
    BloodPressure: float = Field(ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(ge=0, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(ge=0, le=70, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(
        ge=0, le=3, description="Diabetes pedigree function"
    )
    Age: int = Field(ge=0, le=120, description="Age in years")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: str  # "diabetic" or "non-diabetic"
    probability: float
    confidence: float  # percentage


class ModelInfoResponse(BaseModel):
    """Response schema for model-info endpoint."""

    best_model: str
    selected_features: list[str]
    metrics: dict
    baseline_comparison: dict
    ga_comparison: dict
