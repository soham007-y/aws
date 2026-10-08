#!/usr/bin/env python3
"""
03_modeling.py - Baseline Supervised XGBoost Model with Class Weighting
Kaggle Credit Card Fraud Detection
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
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
    """
    # 1. Check if preprocessed .npz already exists
    if preprocessed_path and preprocessed_path.exists():
        print(f"Loading preprocessed dataset from: {preprocessed_path}")
        npz = np.load(preprocessed_path, allow_pickle=True)
        return {
            "X_train": npz["X_train_resampled"],
            "y_train": npz["y_train_resampled"],
            "X_test": npz["X_later_scaled"],
            "y_test": npz["y_later"],
            "scale_pos_weight": float(npz["scale_pos_weight"][0]),
        }

    # 2. Check default preprocessed location
    default_npz = Path("data") / "processed" / "preprocessed_data.npz"
    if default_npz.exists():
        print(f"Loading preprocessed dataset from default path: {default_npz}")
        npz = np.load(default_npz, allow_pickle=True)
        return {
            "X_train": npz["X_train_resampled"],
            "y_train": npz["y_train_resampled"],
            "X_test": npz["X_later_scaled"],
            "y_test": npz["y_later"],
            "scale_pos_weight": float(npz["scale_pos_weight"][0]),
        }

    # 3. Otherwise preprocess from raw dataset
    raw_path = data_path if data_path else (find_dataset() if find_dataset else None)
    if raw_path and raw_path.exists():
        print(f"Preprocessed cache not found. Running preprocessing on: {raw_path}")
        result = preprocess_data(data_path=raw_path)
        return {
            "X_train": result["X_train_resampled"],
            "y_train": result["y_train_resampled"],
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


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Builds and trains an XGBoost classifier with cost-sensitive class weighting.
    """
    print("\n" + "=" * 80)
    print("Initializing & Training Supervised XGBoost Classifier")
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
    print("Model training complete.")
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluates the model on the chronological 'later_window' (test split)
    to measure generalization and robustness against time drift.
    """
    print("\n" + "=" * 80)
    print("Evaluating Model on Later Window (Time Drift Evaluation)")
    print("=" * 80)

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

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
    return metrics


def save_model(
    model: xgb.XGBClassifier,
    models_dir: Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    """
    Saves trained XGBoost model and metadata to the models/ directory.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save in standard XGBoost JSON format and joblib serialization
    model_json_path = models_dir / "xgboost_baseline.json"
    model.save_model(model_json_path)

    model_joblib_path = models_dir / "xgboost_baseline.joblib"
    joblib.dump(model, model_joblib_path)

    if metadata:
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n[Saved Model] Artifacts stored in: {models_dir}")
    print(f"  JSON format:   {model_json_path}")
    print(f"  Joblib format: {model_joblib_path}")
    return model_json_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Baseline Supervised XGBoost Model on Credit Card Fraud Dataset"
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
        "--n-estimators",
        type=int,
        default=150,
        help="Number of gradient boosted trees (default: 150)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum tree depth (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
        help="Boosting learning rate (default: 0.08)",
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

    model = train_xgboost(
        X_train=data["X_train"],
        y_train=data["y_train"],
        scale_pos_weight=data["scale_pos_weight"],
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )

    metrics = evaluate_model(
        model=model,
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    metadata = {
        "model_type": "XGBClassifier",
        "scale_pos_weight": data["scale_pos_weight"],
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "metrics": metrics,
    }

    save_model(model=model, models_dir=models_path, metadata=metadata)


if __name__ == "__main__":
    main()
