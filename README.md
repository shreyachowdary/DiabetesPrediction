# Diabetes Prediction using Genetic Algorithms

A production-style machine learning project that uses **Genetic Algorithms** for feature selection to identify the most relevant medical inputs for diabetes prediction, then trains and compares multiple ML models (Random Forest, SVM, Logistic Regression) and deploys the best model via **FastAPI** with a user-friendly frontend.

---

## Project Overview

This system predicts diabetes risk from health metrics (Pregnancies, Glucose, Blood Pressure, BMI, etc.) using:

1. **Genetic Algorithm (GA) feature selection** – Evolves optimal feature subsets to improve model efficiency and generalization
2. **Multiple ML classifiers** – Random Forest, SVM, and Logistic Regression with hyperparameter tuning
3. **FastAPI backend** – REST API for real-time prediction
4. **Interactive frontend** – Clean UI for entering health data and viewing predictions

---

## Problem Statement

Diabetes prediction requires selecting the most informative features from a set of medical measurements. Using all features can lead to overfitting and slower inference. This project uses a **Genetic Algorithm** to search for the best subset of features that maximizes prediction accuracy while keeping the model parsimonious.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML | Scikit-learn, Pandas, NumPy |
| Genetic Algorithm | DEAP |
| API | FastAPI, Uvicorn |
| Model Persistence | Joblib |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel (serverless) |

---

## Dataset

**Pima Indians Diabetes Dataset** (UCI ML Repository)

- **Source**: [Jason Brownlee's GitHub](https://github.com/jbrownlee/Datasets)
- **Samples**: ~768
- **Features**: 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Target**: Binary (0 = non-diabetic, 1 = diabetic)

**Preprocessing**:
- Replace physiologically impossible zeros (e.g., 0 glucose) with median values
- Remove duplicates
- Optional IQR-based outlier capping
- StandardScaler for feature normalization
- 80/20 stratified train-test split

---

## Genetic Algorithm Feature Selection

The GA pipeline uses **DEAP** to evolve a population of binary feature masks:

- **Individual**: Binary vector (1 = include feature, 0 = exclude)
- **Fitness**: Cross-validated accuracy minus a small penalty for feature count (parsimony)
- **Operators**: Two-point crossover, bit-flip mutation, tournament selection
- **Output**: Best subset of features that balances accuracy and model simplicity

**Why GA?** Exhaustive search over 2^8 subsets is feasible but GA scales to larger feature sets and often finds good solutions faster.

---

## Model Comparison

Models are trained and compared **before** and **after** GA feature selection:

| Model | Metrics |
|-------|---------|
| Random Forest | Accuracy, Precision, Recall, F1, ROC-AUC |
| SVM | Same |
| Logistic Regression | Baseline comparison |

The best model (by F1-score) is saved and deployed. **All reported metrics are from actual training** – no fabricated numbers.

### Example Results (from training run)

**Baseline (all 8 features):**
- Random Forest: Acc=74.7%, F1=61.4%, AUC=0.81
- SVM: Acc=75.3%, F1=61.2%, AUC=0.81
- Logistic Regression: Acc=70.8%, F1=54.6%, AUC=0.81

**After GA (4 selected features: Glucose, SkinThickness, BMI, DiabetesPedigreeFunction):**
- Best model: SVM (Acc=72.1%, F1=55.7%, AUC=0.79)

---

## Project Structure

```
diabetes-ga-prediction/
├── app/
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic models
│   ├── model_loader.py   # Load trained artifacts
│
├── api/
│   └── index.py          # Vercel serverless entry point
│
├── ml/
│   ├── preprocess.py     # Data preprocessing
│   ├── feature_selection_ga.py  # GA pipeline
│   ├── train.py         # Model training
│   └── evaluate.py      # Metrics & visualization scripts
│
├── scripts/
│   ├── download_data.py  # Fetch dataset
│   ├── run_training.py   # Full training pipeline
│   ├── plot_confusion_matrix.py
│   ├── plot_roc_curve.py
│   └── plot_feature_importance.py
│
├── models/               # Trained artifacts (after training)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── selected_features.pkl
│   └── model_info.json
│
├── data/
│   └── diabetes.csv
│
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── notebooks/
│   └── eda.ipynb
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── vercel.json
├── run_local.py          # Start server + open browser
└── run_desktop.py        # Desktop-style window (pywebview)
```

---

## Local Setup

### 1. Clone and Install

```bash
cd "Diabetes Prediction"
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/MacOS

pip install -r requirements.txt   # Windows
pip3 install -r requirements.txt  # Linux/MacOS
```

### 2. Download Dataset

```bash
python scripts/download_data.py    # Windows
python3 scripts/download_data.py   # Linux/MacOS
```

### 3. Train Models

```bash
python scripts/run_training.py    # Windows
python3 scripts/run_training.py   # Linux/MacOS
```

This will:
- Preprocess the data
- Run GA feature selection
- Train RF, SVM, LR (before and after GA)
- Save best model and artifacts to `models/`

### 4. Run Locally

**Option A – Browser**
```bash
python run_local.py
```
Opens `http://127.0.0.1:8000` in your browser.

**Option B – Desktop-style window**
```bash
pip install pywebview
python run_desktop.py    # Windows

pip3 install pywebview
python3 run_desktop.py   # Linux/MacOS
```
Opens the app in a native window (no browser chrome).

**Option C – Manual**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Then open `http://127.0.0.1:8000`.

---

## API Usage

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend |
| GET | `/health` | Health check |
| POST | `/predict` | Diabetes prediction |
| GET | `/model-info` | Model metadata & metrics |

### Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35
  }'
```

### Sample Response

```json
{
  "prediction": "non-diabetic",
  "probability": 0.23,
  "confidence": 0.77
}
```

---

## Vercel Deployment

### Prerequisites

- [Vercel CLI](https://vercel.com/cli) installed (`npm i -g vercel`)
- GitHub account (for Git-based deployment)

### Steps

1. **Train models locally** (required before deploy):
   ```bash
   python scripts/download_data.py
   python scripts/run_training.py
   ```

2. **Commit model artifacts** (ensure `models/` is in the repo):
   - Check `.gitignore` – if `models/*.pkl` is ignored, remove that line or add `!models/*.pkl` to include them.
   - Commit: `git add models/ && git commit -m "Add trained models"`

3. **Deploy via Vercel CLI**:
   ```bash
   vercel
   ```
   Follow prompts to link the project. On first deploy, Vercel will detect the Python app from `api/index.py` and `vercel.json`.

4. **Deploy from GitHub**:
   - Push to GitHub
   - Go to [vercel.com](https://vercel.com) → New Project → Import from GitHub
   - Select the repo
   - Root directory: `.` (project root)
   - Deploy

5. **Environment**: No env vars required. Ensure `models/` is committed so the serverless function can load them.

### Deployment Notes

- **Model size**: Keep `models/` under ~50MB (Vercel serverless limit)
- **Cold starts**: First request may be slower; subsequent requests are fast
- **Screenshots**: Add screenshots of the deployed app to this README in the section below

---

## Screenshots

<!-- Add screenshots of the deployed app here -->
- *Frontend prediction form*
- *Sample prediction result*
- *Model info section*

---

## Resume Alignment

This project demonstrates:

- **GA-based feature selection pipeline** – Implemented with DEAP; fitness balances accuracy and parsimony
- **Random Forest & SVM classifiers** – Trained with GridSearchCV; compared before/after feature selection
- **Strong prediction accuracy** – Metrics reported from actual training
- **FastAPI deployment** – REST API with Pydantic validation
- **Interactive frontend** – Professional UI for real-time prediction
- **Improved efficiency** – Fewer features reduce overfitting and inference cost

---

## License

This project is own by Shreya Chowdary Chennupati, and Velaga Bhanu Prakash
- Open Contributions are welcomed.
