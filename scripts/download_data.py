"""
Download the Pima Indians Diabetes Dataset.

Source: UCI ML Repository / Jason Brownlee's GitHub
"""

import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
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


def download_diabetes_data(output_dir: Path) -> Path:
    """Download dataset and save as CSV with headers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diabetes.csv"
    
    print(f"Downloading from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, output_path)
    
    # Add header row
    with open(output_path, "r") as f:
        content = f.read()
    
    header = ",".join(COLUMNS) + "\n"
    with open(output_path, "w") as f:
        f.write(header + content)
    
    print(f"Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    download_diabetes_data(data_dir)
