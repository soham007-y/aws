# Financial Fraud Detection & Concept Drift Monitoring Pipeline

An end-to-end, production-grade machine learning architecture for extreme credit card fraud detection (~0.17% minority prevalence), cost-sensitive decision optimization, and temporal concept drift monitoring.

---

## Author Information

- **Author**: Soham Sachin Yadav
- **Registration Number**: 26BIT0259

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
   [Robust & Standard Scalers]                     [Temporal Evaluation Set]
              │                                               │
   Controlled SMOTE (0.10)                                    │
   + Class Weights (scale_pos_weight)                         │
              │                                               │
      ┌───────┴────────────────────────┐                      │
      ▼                                ▼                      │
Supervised XGBoost           Unsupervised Isolation           │
[Hist Gradient Boosting]     Forest (Normal Txns Only)        │
      │                                │                      │
      └───────┬────────────────────────┘                      │
              ▼                                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 04_evaluate.py: PR-AUC & Business Loss Threshold Optimizer   │
   │  * Cost: $10/FP (friction) vs $Amount/FN (fraud loss)       │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 05_drift_monitor.py: Kolmogorov-Smirnov (KS) Drift Alerts    │
   │  * Retraining trigger: KS > 0.05 on > 3 critical features   │
   └──────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```text
├── 01_eda.py              # Exploratory data analysis & chronological partition
├── 02_preprocess.py       # Scalers, controlled SMOTE, & class weighting
├── 03_modeling.py         # Supervised XGBoost & Unsupervised Isolation Forest
├── 04_evaluate.py         # PR-AUC & custom business loss threshold sweep
├── 05_drift_monitor.py    # Two-sample KS test drift monitor & alert engine
├── requirements.txt       # Production dependencies
├── WRITEUP.md             # In-depth architectural & theoretical methodology
├── README.md              # Documentation & reproduction guide
└── models/                # Serialized model artifacts, weights & JSON logs
    ├── xgboost_baseline.joblib
    ├── xgboost_baseline.json
    ├── isolation_forest.joblib
    ├── model_metadata.json
    ├── isolation_forest_metadata.json
    ├── evaluation_results.json
    └── drift_report.json
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Kaggle Credit Card Fraud dataset: [`creditcard.csv`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 2. Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/soham007-y/aws.git
cd aws

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Place Dataset
Place `creditcard.csv` in the root workspace directory or inside a `data/` folder:
```bash
# Verify placement
ls -l creditcard.csv
# or
mkdir -p data && mv /path/to/creditcard.csv data/creditcard.csv
```

---

## Pipeline Execution & Reproduction

Run each stage sequentially to reproduce the end-to-end evaluation:

### Step 1: Exploratory Data Analysis & Imbalance Audit
Calculates exact class imbalance, distribution summaries, and chronological window statistics:
```bash
python3 01_eda.py --data-path creditcard.csv
```

### Step 2: Preprocessing, Scaling & Controlled SMOTE
Applies `RobustScaler` on `Amount`, `StandardScaler` on `Time`, controlled SMOTE (`sampling_strategy=0.10`), and calculates XGBoost `scale_pos_weight`:
```bash
python3 02_preprocess.py --data-path creditcard.csv --output-dir data/processed
```

### Step 3: Model Training (Supervised XGBoost + Unsupervised Isolation Forest)
Trains both models and serializes artifacts into the `models/` directory:
```bash
python3 03_modeling.py --data-path creditcard.csv --models-dir models --pipeline all
```
*Optional pipeline flags:*
- `--pipeline xgboost`: Train only the supervised XGBoost classifier.
- `--pipeline isolation_forest`: Train only the unsupervised Isolation Forest.

### Step 4: PR-AUC Evaluation & Business Cost Threshold Optimization
Evaluates PR-AUC on the chronological later window and executes a threshold sweep minimizing total dollar loss ($\$10$ FP friction vs. $\$ \text{Amount}$ FN fraud loss):
```bash
python3 04_evaluate.py --data-path creditcard.csv --models-dir models --cost-fp 10.0
```

### Step 5: Kolmogorov-Smirnov Feature Drift Monitoring & Retraining Alerts
Runs two-sample KS statistical tests between training and later operational windows, testing for distribution divergence across critical features (`V14`, `V17`, `V12`, `V10`, `V4`, `V11`, `V2`, `Amount`):
```bash
python3 05_drift_monitor.py --data-path creditcard.csv --drift-threshold 0.05 --max-critical-drift 3
```

---

## Key Metrics & Configuration Summary

| Component | Strategy / Hyperparameter | Production Value |
| :--- | :--- | :--- |
| **Chronological Split** | Temporal partition (no shuffle) | First 70% Train / Last 30% Evaluation |
| **Resampling** | Controlled SMOTE | `sampling_strategy = 0.10` |
| **Class Weighting** | Cost-sensitive gradient scaling | `scale_pos_weight = N_neg / N_pos` |
| **Supervised Model** | Histogram Gradient Boosted Trees | `XGBClassifier(eval_metric='aucpr', max_depth=5)` |
| **Unsupervised Model** | Isolation Forest (inliers only) | `IsolationForest(contamination=0.0017, max_samples=256)` |
| **Primary Metric** | Precision-Recall AUC (PR-AUC) | Isolates minority class without TN dilution |
| **Business Cost Function** | Minimization of operational cost | $\text{Cost} = 10 \times \text{FP} + \sum \text{Amount}_{\text{FN}}$ |
| **Drift Monitoring** | Two-Sample Kolmogorov-Smirnov Test | Trigger alert when $\text{KS} > 0.05$ on $> 3$ critical features |
