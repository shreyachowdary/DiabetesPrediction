"""
Data preprocessing module for the Pima Indians Diabetes Dataset.

Handles missing values, duplicates, scaling, and train-test splitting.
Each step is modular and documented for maintainability.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure module-level logger
logger = logging.getLogger(__name__)

# Pima dataset uses 0 to denote missing values for certain features
# (glucose, blood pressure, skin thickness, insulin, BMI)
FEATURES_WITH_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Expected columns in the raw dataset
EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Load the diabetes dataset from CSV.
    Raises FileNotFoundError if the file does not exist.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path.name}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure the dataset has the expected structure.
    Some versions of the dataset use different column names.
    """
    # Handle alternate column naming (e.g., lowercase, underscores)
    if "Outcome" not in df.columns and "outcome" in df.columns:
        df.rename(columns={"outcome": "Outcome"}, inplace=True)
    
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            # Try case-insensitive match
            match = next((c for c in df.columns if c.lower() == col.lower()), None)
            if match:
                df.rename(columns={match: col}, inplace=True)
            else:
                raise ValueError(f"Expected column '{col}' not found. Got: {list(df.columns)}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace physiologically impossible zeros with median values.
    
    In the Pima dataset, zero is used as a placeholder for missing data
    in Glucose, BloodPressure, SkinThickness, Insulin, and BMI.
    These cannot be zero in real patients (e.g., no one has 0 glucose).
    """
    df = df.copy()
    
    for col in FEATURES_WITH_ZERO_AS_MISSING:
        if col not in df.columns:
            continue
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            median_val = df[df[col] != 0][col].median()
            df.loc[df[col] == 0, col] = median_val
            logger.info(f"Replaced {zero_count} zeros in '{col}' with median {median_val:.2f}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows to prevent data leakage and overfitting.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    return df


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Optional outlier handling using IQR method.
    Caps extreme values rather than removing rows to preserve sample size.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "Outcome"]
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        # Cap values instead of dropping rows
        df[col] = df[col].clip(lower=lower, upper=upper)
    
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into feature matrix X and target vector y.
    """
    feature_cols = [c for c in EXPECTED_COLUMNS if c != "Outcome"]
    X = df[feature_cols].copy()
    y = df["Outcome"].copy()
    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply StandardScaler to normalize features.
    Fit on train only to avoid data leakage; transform both train and test.
    Returns numpy arrays for compatibility with sklearn models.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    data_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    handle_outliers_flag: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """
    Full preprocessing pipeline.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = load_raw_data(data_path)
    validate_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    
    if handle_outliers_flag:
        df = handle_outliers(df)
    
    X, y = prepare_features_and_target(df)
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    logger.info(
        f"Preprocessing complete: train={len(y_train)}, test={len(y_test)}, "
        f"features={len(feature_names)}"
    )
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, feature_names
