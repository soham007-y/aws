#!/usr/bin/env python3
"""
03_modeling.py - Fraud Detection Modeling Pipelines
1. Supervised Pipeline: Cost-Sensitive XGBoost Classifier with Class Weighting
2. Unsupervised Pipeline: Isolation Forest trained strictly on normal (non-fraud) transactions
Includes evaluation and baseline comparison on the chronological 'later_window' (time drift).
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import xgboost as xgb
import joblib

# Import preprocessing pipeline from 02_preprocess dynamically (filename starts with digit)
import importlib.util

preprocess_module_path = Path(__file__).parent / "02_preprocess.py"
if preprocess_module_path.exists():
    spec = importlib.util.spec_from_file_location("preprocess_mod", preprocess_module_path)
    preprocess_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess_mod)
    preprocess_data = preprocess_mod.preprocess_data
    find_dataset = preprocess_mod.find_dataset
else:
    preprocess_data = None
    find_dataset = None


def load_training_data(
    data_path: Path | None = None,
    preprocessed_path: Path | None = None,
) -> dict[str, np.ndarray | float | list[str]]:
    """
    Loads preprocessed splits from compressed .npz archive,
    or falls back to preprocessing the raw creditcard.csv dataset.
    Extracts both the balanced training set for supervised learning
    and strictly normal (non-fraud) transactions for unsupervised learning.
    """
    # 1. Check specified preprocessed .npz
    if preprocessed_path and preprocessed_path.exists():
        print(f"Loading preprocessed dataset from: {preprocessed_path}")
        npz = np.load(preprocessed_path, allow_pickle=True)
        X_train = npz["X_train_resampled"]
        y_train = npz["y_train_resampled"]
        X_test = npz["X_later_scaled"]
        y_test = npz["y_later"]
        scale_pos_weight = float(npz["scale_pos_weight"][0])
        # In SMOTE resampled data, Class 0 instances are strictly the original normal transactions
        X_train_normal = X_train[y_train == 0]
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_train_normal": X_train_normal,
            "X_test": X_test,
            "y_test": y_test,
            "scale_pos_weight": scale_pos_weight,
        }

    # 2. Check default preprocessed location
    default_npz = Path("data") / "processed" / "preprocessed_data.npz"
    if default_npz.exists():
        print(f"Loading preprocessed dataset from default path: {default_npz}")
        npz = np.load(default_npz, allow_pickle=True)
        X_train = npz["X_train_resampled"]
        y_train = npz["y_train_resampled"]
        X_test = npz["X_later_scaled"]
        y_test = npz["y_later"]
        scale_pos_weight = float(npz["scale_pos_weight"][0])
        X_train_normal = X_train[y_train == 0]
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_train_normal": X_train_normal,
            "X_test": X_test,
            "y_test": y_test,
            "scale_pos_weight": scale_pos_weight,
        }

    # 3. Preprocess from raw dataset
    raw_path = data_path if data_path else (find_dataset() if find_dataset else None)
    if raw_path and raw_path.exists():
        print(f"Preprocessed cache not found. Running preprocessing on: {raw_path}")
        result = preprocess_data(data_path=raw_path)
        X_train = result["X_train_resampled"]
        y_train = result["y_train_resampled"]
        X_train_normal = X_train[y_train == 0]
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_train_normal": X_train_normal,
            "X_test": result["X_later_scaled"],
            "y_test": result["y_later"],
            "scale_pos_weight": float(result["scale_pos_weight"]),
            "feature_names": result.get("feature_names", []),
        }

    raise FileNotFoundError(
        "Neither preprocessed splits (data/processed/preprocessed_data.npz) "
        "nor raw dataset (creditcard.csv) could be located. "
        "Please provide --preprocessed-path or --data-path."
    )


# ==============================================================================
# PIPELINE 1: SUPERVISED XGBOOST CLASSIFIER
# ==============================================================================


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Builds and trains an XGBoost classifier with cost-sensitive class weighting."""
    print("\n" + "=" * 80)
    print("PIPELINE 1: Initializing & Training Supervised XGBoost Classifier")
    print("=" * 80)
    print(f"Training samples:     {X_train.shape[0]:,}")
    print(f"Feature count:        {X_train.shape[1]}")
    print(f"Minority class weight (scale_pos_weight): {scale_pos_weight:.4f}")
    print(f"Hyperparameters:      n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("XGBoost model training complete.")
    return model


def evaluate_xgboost(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluates the XGBoost model on the chronological 'later_window' (test split)."""
    print("\n" + "-" * 80)
    print("Evaluating XGBoost on Later Window (Time Drift Evaluation)")
    print("-" * 80)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Test Samples:       {len(y_test):,}")
    print(f"Actual Frauds:      {int(y_test.sum()):,}")
    print(f"Predicted Frauds:   {int(y_pred.sum()):,}")
    print(f"\nArea Under PR Curve (PR-AUC):   {pr_auc:.4f}  (Key Metric for Extreme Imbalance)")
    print(f"Area Under ROC Curve (ROC-AUC): {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, target_names=["Legitimate (0)", "Fraud (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"  True Negatives (TN):  {tn:,}")
    print(f"  False Positives (FP): {fp:,}")
    print(f"  False Negatives (FN): {fn:,}")
    print(f"  True Positives (TP):  {tp:,}")

    return {
        "model_name": "Supervised XGBoost",
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def save_xgboost_model(
    model: xgb.XGBClassifier,
    models_dir: Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Saves trained XGBoost model and metadata to models/ directory."""
    models_dir.mkdir(parents=True, exist_ok=True)

    model_json_path = models_dir / "xgboost_baseline.json"
    model.save_model(model_json_path)

    model_joblib_path = models_dir / "xgboost_baseline.joblib"
    joblib.dump(model, model_joblib_path)

    if metadata:
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n[Saved Model] XGBoost artifacts stored in: {models_dir}")
    print(f"  JSON format:   {model_json_path}")
    print(f"  Joblib format: {model_joblib_path}")
    return model_json_path


# ==============================================================================
# PIPELINE 2: UNSUPERVISED ISOLATION FOREST MODEL
# ==============================================================================


def train_isolation_forest(
    X_train_normal: np.ndarray,
    n_estimators: int = 150,
    contamination: float = 0.0017,
    max_samples: int = 256,
    random_state: int = 42,
) -> IsolationForest:
    """
    Builds and trains an unsupervised Isolation Forest model strictly on
    normal (non-fraud) transactions from the training set.

    Rationale:
    - In semi-supervised anomaly detection, the model learns the geometric
      manifold of normal cardholder behavior without seeing fraud labels.
    - At inference time, transactions that isolate in fewer tree splits are
      flagged as anomalous deviations (fraud candidates).
    - contamination is set to the approximate empirical fraud frequency (~0.17%).
    - max_samples=256 bounds tree height to mitigate masking and swamping effects.
    """
    print("\n" + "=" * 80)
    print("PIPELINE 2: Training Unsupervised Isolation Forest (Normal Transactions Only)")
    print("=" * 80)
    print(f"Training samples (strictly normal/non-fraud): {X_train_normal.shape[0]:,}")
    print(f"Feature count:                               {X_train_normal.shape[1]}")
    print(f"Contamination parameter:                     {contamination} (~0.17% expected outliers)")
    print(f"Subsample per tree (max_samples):            {max_samples}")

    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    iso_forest.fit(X_train_normal)
    print("Isolation Forest training complete.")
    return iso_forest


def evaluate_isolation_forest(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluates Isolation Forest on the chronological 'later_window'.
    - Invert decision_function to score anomalies: lower decision_function = higher fraud risk.
    - Convert predictions (-1: anomaly, 1: inlier) to (1: fraud, 0: legitimate).
    """
    print("\n" + "-" * 80)
    print("Evaluating Isolation Forest on Later Window (Time Drift Evaluation)")
    print("-" * 80)

    # Inliers are +1, outliers/anomalies are -1. Map to 0 (normal) and 1 (fraud)
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)

    # decision_function yields signed distance: negative for outliers, positive for inliers
    # Negative distance -> higher anomaly score
    anomaly_scores = -model.decision_function(X_test)

    pr_auc = average_precision_score(y_test, anomaly_scores)
    roc_auc = roc_auc_score(y_test, anomaly_scores)

    print(f"Test Samples:       {len(y_test):,}")
    print(f"Actual Frauds:      {int(y_test.sum()):,}")
    print(f"Predicted Frauds:   {int(y_pred.sum()):,}")
    print(f"\nArea Under PR Curve (PR-AUC):   {pr_auc:.4f}  (Unsupervised Anomaly Score)")
    print(f"Area Under ROC Curve (ROC-AUC): {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, target_names=["Legitimate (0)", "Fraud (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"  True Negatives (TN):  {tn:,}")
    print(f"  False Positives (FP): {fp:,}")
    print(f"  False Negatives (FN): {fn:,}")
    print(f"  True Positives (TP):  {tp:,}")

    return {
        "model_name": "Unsupervised Isolation Forest",
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def save_isolation_forest_model(
    model: IsolationForest,
    models_dir: Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Saves trained Isolation Forest model and metadata to models/ directory."""
    models_dir.mkdir(parents=True, exist_ok=True)

    model_joblib_path = models_dir / "isolation_forest.joblib"
    joblib.dump(model, model_joblib_path)

    if metadata:
        metadata_path = models_dir / "isolation_forest_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n[Saved Model] Isolation Forest artifact stored in: {model_joblib_path}")
    return model_joblib_path


# ==============================================================================
# BASELINE COMPARISON SUMMARY
# ==============================================================================


def display_baseline_comparison(
    xgb_metrics: dict[str, float],
    iso_metrics: dict[str, float],
) -> None:
    """Prints a comparative summary table contrasting supervised vs unsupervised models."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON: Supervised XGBoost vs Unsupervised Isolation Forest")
    print("=" * 80)
    headers = f"{'Metric':<25} | {'Supervised XGBoost':<22} | {'Isolation Forest':<22}"
    print(headers)
    print("-" * len(headers))
    print(f"{'PR-AUC (Key Metric)':<25} | {xgb_metrics['pr_auc']:<22.4f} | {iso_metrics['pr_auc']:<22.4f}")
    print(f"{'ROC-AUC':<25} | {xgb_metrics['roc_auc']:<22.4f} | {iso_metrics['roc_auc']:<22.4f}")
    print(f"{'True Positives (TP)':<25} | {xgb_metrics['true_positives']:<22,d} | {iso_metrics['true_positives']:<22,d}")
    print(f"{'False Positives (FP)':<25} | {xgb_metrics['false_positives']:<22,d} | {iso_metrics['false_positives']:<22,d}")
    print(f"{'False Negatives (FN)':<25} | {xgb_metrics['false_negatives']:<22,d} | {iso_metrics['false_negatives']:<22,d}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train Fraud Detection Models (Supervised XGBoost & Unsupervised Isolation Forest)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to creditcard.csv dataset",
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        default=None,
        help="Path to preprocessed_data.npz",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Target directory to save model artifacts (default: models)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["all", "xgboost", "isolation_forest"],
        default="all",
        help="Which pipeline(s) to train: all, xgboost, or isolation_forest (default: all)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    models_path = Path(args.models_dir)
    data_path = Path(args.data_path) if args.data_path else None
    prep_path = Path(args.preprocessed_path) if args.preprocessed_path else None

    try:
        data = load_training_data(data_path=data_path, preprocessed_path=prep_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    xgb_metrics = None
    iso_metrics = None

    # Pipeline 1: Supervised XGBoost
    if args.pipeline in ["all", "xgboost"]:
        xgb_model = train_xgboost(
            X_train=data["X_train"],
            y_train=data["y_train"],
            scale_pos_weight=data["scale_pos_weight"],
            random_state=args.random_state,
        )
        xgb_metrics = evaluate_xgboost(
            model=xgb_model,
            X_test=data["X_test"],
            y_test=data["y_test"],
        )
        save_xgboost_model(
            model=xgb_model,
            models_dir=models_path,
            metadata={
                "model_type": "XGBClassifier",
                "scale_pos_weight": data["scale_pos_weight"],
                "metrics": xgb_metrics,
            },
        )

    # Pipeline 2: Unsupervised Isolation Forest (strictly normal transactions)
    if args.pipeline in ["all", "isolation_forest"]:
        iso_model = train_isolation_forest(
            X_train_normal=data["X_train_normal"],
            contamination=0.0017,
            max_samples=256,
            random_state=args.random_state,
        )
        iso_metrics = evaluate_isolation_forest(
            model=iso_model,
            X_test=data["X_test"],
            y_test=data["y_test"],
        )
        save_isolation_forest_model(
            model=iso_model,
            models_dir=models_path,
            metadata={
                "model_type": "IsolationForest",
                "trained_on": "strictly_normal_transactions",
                "contamination": 0.0017,
                "metrics": iso_metrics,
            },
        )

    # Baseline Comparison
    if xgb_metrics and iso_metrics:
        display_baseline_comparison(xgb_metrics, iso_metrics)


if __name__ == "__main__":
    main()
