# Financial Fraud Detection & Concept Drift Monitoring Pipeline

An end-to-end, production-grade machine learning system for extreme credit card fraud detection (~0.17% minority prevalence), cost-sensitive decision optimization, and temporal concept drift monitoring.

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │   creditcard.csv       │
                          │   (284,807 txns)       │
                          └───────────┬────────────┘
                                      │
                         Chronological Split (70 / 30)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   Training Window (First 70%)                     Later Window (Last 30%)
   [199,364 samples]                               [85,443 samples]
              │                                               │
   Controlled SMOTE (0.10)                                    │
   + Class Weights (scale_pos_weight)                         │
              │                                               │
      ┌───────┴────────────────────────┐                      │
      ▼                                ▼                      │
Supervised XGBoost           Unsupervised Isolation           │
[Cost-Sensitive Classifier]  Forest (Normal Txns Only)        │
      │                                │                      │
      └───────┬────────────────────────┘                      │
              ▼                                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 04_evaluate.py: PR-AUC & Business Cost Threshold Optimizer   │
   │  * Custom Cost: $10/FP (friction) vs $Amount/FN (fraud loss) │
   │  * Sweeps cutoffs (0.01 to 0.99) to find minimum dollar loss │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 05_drift_monitor.py: Kolmogorov-Smirnov (KS) Drift Alerts    │
   │  * Two-sample KS test across V1-V28 and Amount               │
   │  * Prediction confidence distribution shift check            │
   │  * Retraining alert triggered if > 3 critical features drift │
   └──────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```text
├── 01_eda.py              # Exploratory data analysis & chronological partition
├── 02_preprocess.py       # Scalers, controlled SMOTE, & class weighting
├── 03_modeling.py         # Supervised XGBoost & Unsupervised Isolation Forest
├── 04_evaluate.py         # PR-AUC evaluation & business cost threshold sweep
├── 05_drift_monitor.py    # Two-sample KS test drift monitor & alert engine
├── requirements.txt       # Production dependencies
├── WRITEUP.md             # In-depth architectural & theoretical technical report
├── README.md              # Documentation & reproduction guide
├── data/                  # Generated chronological splits & preprocessed data (git-ignored)
│   ├── train_window.csv
│   ├── later_window.csv
│   ├── train_preprocessed.csv
│   └── val_preprocessed.csv
└── models/                # Serialized model artifacts, weights & JSON reports
    ├── xgboost_baseline.joblib
    ├── xgboost_baseline.json
    ├── isolation_forest.joblib
    ├── evaluation_results.json
    └── drift_report.json
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Kaggle Credit Card Fraud dataset: [`creditcard.csv`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 2. Create & Activate Virtual Environment
```bash
# Clone the repository
git clone https://github.com/soham007-y/aws.git
cd aws

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Dataset Placement
Place `creditcard.csv` directly in the root project directory or inside the `data/` folder:
```bash
# Verify dataset placement
ls -lh creditcard.csv
# or
ls -lh data/creditcard.csv
```

---

## Step-by-Step Reproduction Guide

Execute the scripts sequentially from `01_eda.py` through `05_drift_monitor.py` to reproduce the entire pipeline:

### Step 1: Exploratory Data Analysis & Chronological Split
Performs dataset audit, computes exact class imbalance (~1:578 ratio), and splits data chronologically (first 70% training, last 30% operational) without shuffling:
```bash
python 01_eda.py
```
*Output artifacts:* `data/train_window.csv` (199,364 rows) and `data/later_window.csv` (85,443 rows).

### Step 2: Feature Preprocessing, Scaling & Hybrid SMOTE
Scales numerical features (`RobustScaler` for `Amount`, `StandardScaler` for `Time`), applies controlled SMOTE (`sampling_strategy=0.10`) on the training split, and computes balanced class weights:
```bash
python 02_preprocess.py
```
*Output artifacts:* `data/train_preprocessed.csv`, `data/val_preprocessed.csv`, and `data/preprocessed_data.npz`.

### Step 3: Model Training (Supervised XGBoost & Isolation Forest)
Trains the cost-sensitive Supervised XGBoost classifier and the Unsupervised Isolation Forest (strictly on normal transactions):
```bash
python 03_modeling.py
```
*Output artifacts:* `models/xgboost_baseline.joblib`, `models/xgboost_baseline.json`, and `models/isolation_forest.joblib`.

### Step 4: PR-AUC Evaluation & Business Cost Threshold Optimization
Evaluates Precision-Recall AUC (prioritized over ROC-AUC) on the validation split and runs a decision threshold sweep (0.01 to 0.99) minimizing total business financial impact ($10 FP friction vs. actual transaction Amount FN loss):
```bash
python 04_evaluate.py
```
*Output:* PR-AUC scores, optimal threshold cutoff (0.80 for XGBoost), and financial savings ($600 savings vs default 0.50 cutoff). Saves `models/evaluation_results.json`.

### Step 5: Feature & Prediction Confidence Drift Monitoring
Runs the two-sample Kolmogorov-Smirnov (KS) test across all numeric features (`V1` through `V28` and `Amount`), tests for prediction confidence distribution shift, logs a summary table, and triggers an automated retraining alert:
```bash
python 05_drift_monitor.py
```
*Output:* Drift summary table, confidence distribution metrics, and automated retraining alert. Saves `models/drift_report.json`.

---

## Key Pipeline Results Summary

| Metric / Pipeline Step | Value / Configuration |
| :--- | :--- |
| **Chronological Split** | 70% Training Window / 30% Later Window |
| **Minority Class Imbalance** | 0.1727% (1 fraud in every ~579 transactions) |
| **Hybrid Resampling Strategy** | SMOTE (`sampling_strategy=0.10`) + `scale_pos_weight=10.0` |
| **Supervised XGBoost PR-AUC** | **0.6556** (ROC-AUC: 0.9627) |
| **Unsupervised Isolation Forest PR-AUC** | **0.0146** (ROC-AUC: 0.9554) |
| **Optimal XGBoost Threshold** | **0.80** (Min Total Business Cost: $3,823.96 vs $4,423.96 @ 0.50) |
| **Drift Alert Trigger** | **FLAGGED FOR AUTOMATIC RETRAINING** (8 critical features drifted, confidence shifted) |

---

## Author

- **Name**: Soham Sachin Yadav
- **Registration Number**: 26BIT0259
