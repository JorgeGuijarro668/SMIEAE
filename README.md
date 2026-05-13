# SMIEAE — Stress and Anxiety Monitoring via Wearable Devices

This project develops machine learning models to predict and classify **stress and anxiety levels** in university students using physiological data collected from wearable fitness trackers (Fitbit).

## Project Overview

Data from 32 participants across multiple universities was collected over the period February–June 2025. Each participant wore a wearable device during the study period and answered daily questionnaires reporting their perceived stress and anxiety levels. The project combines sensor-derived features with daily survey labels to build predictive models.

**Sensor modalities used:**
- Heart Rate Variability (HRV)
- Blood Oxygen Saturation (SpO2)
- Sleep Score
- Respiratory Rate
- Physical Activity

## Repository Structure

```
SMIEAE/
├── masterfiles/
│   └── all_users_all_metrics.csv   # Final consolidated dataset (all users, all metrics)
├── EDA.ipynb                        # Exploratory Data Analysis — part 1
├── EDA_2.ipynb                      # Exploratory Data Analysis — part 2
├── stress_anxiety_prediction_unified.ipynb      # Regression models for stress/anxiety prediction
├── stress_anxiety_classification_unified.ipynb  # Classification models for stress/anxiety levels
├── requirements.txt                 # Python dependencies
└── README.md
```

## Notebooks

| Notebook | Description |
|---|---|
| `EDA.ipynb` | Initial exploration of the dataset: distributions, correlations, missing values, temporal trends |
| `EDA_2.ipynb` | Extended analysis: cross-user comparisons, university-level aggregations, feature relationships |
| `stress_anxiety_prediction_unified.ipynb` | Regression approach — predicts continuous stress/anxiety scores using Linear Regression, Lasso, ElasticNet, XGBoost, LightGBM, and LSTM |
| `stress_anxiety_classification_unified.ipynb` | Classification approach — predicts stress/anxiety categories using Random Forest, Logistic Regression, KNN, SVM; handles class imbalance with SMOTE |

## Dataset

The final dataset (`masterfiles/all_users_all_metrics.csv`) contains daily-level aggregated features for all participants, merged with their self-reported stress and anxiety scores from daily questionnaires.

## Setup

### Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/JorgeGuijarro668/SMIEAE.git
cd SMIEAE

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Open any of the `.ipynb` files from the Jupyter interface.

## Models

### Prediction (Regression)
- Linear Regression baseline
- Lasso and ElasticNet (regularized linear)
- XGBoost and LightGBM (gradient boosting)
- LSTM (deep learning, time-series)

### Classification
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Class imbalance handled with SMOTE oversampling

### Evaluation
- Regression: MAE, RMSE, R²
- Classification: Accuracy, F1-score, Confusion Matrix
- Cross-validation: TimeSeriesSplit (respects temporal ordering)
