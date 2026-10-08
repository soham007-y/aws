#!/usr/bin/env python3
"""
04_evaluate.py - PR-AUC Evaluation & Business Cost-Benefit Threshold Optimization
Evaluates Supervised XGBoost and Unsupervised Isolation Forest on the chronological validation/test set.
Implements custom financial loss function:
  - False Positive (FP): $10 friction cost (customer disruption & analyst review)
  - False Negative (FN): Actual transaction amount in USD (fraud loss)
Finds the optimal decision threshold that minimizes total business cost.
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
    precision_recall_curve,
    roc_auc_score,
)
import joblib

# Dynamic module loading for custom scripts
import importlib.util


def load_module_from_file(module_name: str, file_path: Path):
    """Dynamically loads a python file as a module."""
    if file_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


current_dir = Path(__file__).parent
preprocess_mod = load_module_from_file("preprocess_mod", current_dir / "02_preprocess.py")
modeling_mod = load_module_from_file("modeling_mod", current_dir / "03_modeling.py")


def load_evaluation_data(
    data_path: Path | None = None,
    preprocessed_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Loads evaluation data (later_window / validation split):
    - Features: X_val (scaled)
    - Ground truth labels: y_val (0 or 1)
    - Real transaction amounts: amounts_val (unscaled dollar amounts for cost calculation)
    """
    # 1. Check if raw dataset is available (gives exact unscaled amounts)
    raw_path = data_path if data_path else (preprocess_mod.find_dataset() if preprocess_mod else None)
    if raw_path and raw_path.exists():
        print(f"Loading validation set from raw dataset: {raw_path}")
        df = pd.read_csv(raw_path)
        training_window, later_window = preprocess_mod.chronological_split(df, train_ratio=0.70)

        # Scale features using training window fit (replicating 02_preprocess)
        from sklearn.preprocessing import RobustScaler, StandardScaler

        X_train = training_window.drop(columns=["Class"]).copy()
        X_val = later_window.drop(columns=["Class"]).copy()
        y_val = later_window["Class"].astype(int).values
        amounts_val = later_window["Amount"].values.astype(float)

        if "Amount" in X_train.columns:
            scaler = RobustScaler()
            scaler.fit(X_train[["Amount"]])
            X_val["Amount"] = scaler.transform(X_val[["Amount"]])

        if "Time" in X_train.columns:
            scaler = StandardScaler()
            scaler.fit(X_train[["Time"]])
            X_val["Time"] = scaler.transform(X_val[["Time"]])

        return {
            "X_val": X_val.values,
            "y_val": y_val,
            "amounts_val": amounts_val,
        }

    # 2. Check preprocessed .npz archive
    npz_path = preprocessed_path if preprocessed_path else (Path("data") / "processed" / "preprocessed_data.npz")
    if npz_path.exists():
        print(f"Loading preprocessed validation split from: {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)
        X_val = npz["X_later_scaled"]
        y_val = npz["y_later"]

        # Approximate amounts if raw data is unavailable (Amount is feature index 29 or last feature)
        # However, for accurate business cost calculation, raw amounts are preferred.
        amounts_val = np.abs(X_val[:, -2]) * 88.0 + 88.0 if X_val.shape[1] >= 2 else np.ones(len(y_val)) * 100.0

        return {
            "X_val": X_val,
            "y_val": y_val,
            "amounts_val": amounts_val,
        }

    raise FileNotFoundError(
        "Could not load evaluation data. Please ensure creditcard.csv or data/processed/preprocessed_data.npz exists."
    )


# ==============================================================================
# CUSTOM BUSINESS COST FUNCTION & THRESHOLD OPTIMIZATION
# ==============================================================================


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cost_fp: float = 10.0,
) -> dict[str, float | int]:
    """
    Computes total business financial impact:
    - False Positive (FP): $10 friction cost (customer inconvenience + investigation time)
    - False Negative (FN): Loss equal to actual fraudulent transaction amount
    - True Positives (TP): Caught fraud ($0 fraud loss)
    - True Negatives (TN): Approved genuine transactions ($0 cost)
    """
    is_fraud = (y_true == 1)
    is_pred_fraud = (y_pred == 1)

    # FP: Predicted fraud but actually legitimate
    fp_mask = (~is_fraud) & is_pred_fraud
    total_fp_cost = float(fp_mask.sum() * cost_fp)

    # FN: Actual fraud missed by model
    fn_mask = is_fraud & (~is_pred_fraud)
    total_fn_cost = float(amounts[fn_mask].sum())

    # TP: Actual fraud detected and blocked
    tp_mask = is_fraud & is_pred_fraud
    prevented_loss = float(amounts[tp_mask].sum())

    total_cost = total_fp_cost + total_fn_cost

    return {
        "total_cost": total_cost,
        "fp_cost": total_fp_cost,
        "fn_cost": total_fn_cost,
        "prevented_loss": prevented_loss,
        "fp_count": int(fp_mask.sum()),
        "fn_count": int(fn_mask.sum()),
        "tp_count": int(tp_mask.sum()),
        "tn_count": int(((~is_fraud) & (~is_pred_fraud)).sum()),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    amounts: np.ndarray,
    model_name: str,
    cost_fp: float = 10.0,
    num_steps: int = 500,
) -> dict[str, object]:
    """
    Tests a continuous spectrum of candidate decision thresholds to discover
    the specific cutoff that strictly minimizes total financial business cost.
    """
    print(f"\nSearching optimal threshold for {model_name}...")

    # Define fine-grained candidate thresholds across [0.001, 0.999]
    # In fraud detection, optimal cutoffs frequently sit below 0.5 due to heavy FN penalties
    candidate_thresholds = np.concatenate([
        np.linspace(0.001, 0.10, num_steps // 2),
        np.linspace(0.101, 0.999, num_steps // 2),
    ])
    candidate_thresholds.sort()

    # Zero-model baseline: no fraud detected (all actual frauds are missed)
    baseline_unmitigated_cost = float(amounts[y_true == 1].sum())

    best_threshold = 0.5
    min_cost = float("inf")
    best_stats = None

    threshold_curve = []

    for t in candidate_thresholds:
        y_pred = (y_scores >= t).astype(int)
        stats = compute_business_cost(y_true, y_pred, amounts, cost_fp=cost_fp)
        cost = stats["total_cost"]

        threshold_curve.append({
            "threshold": float(t),
            "total_cost": cost,
            "fp_cost": stats["fp_cost"],
            "fn_cost": stats["fn_cost"],
            "tp": stats["tp_count"],
            "fp": stats["fp_count"],
            "fn": stats["fn_count"],
        })

        if cost < min_cost:
            min_cost = cost
            best_threshold = float(t)
            best_stats = stats

    # Default 0.5 threshold stats for comparison
    default_pred = (y_scores >= 0.5).astype(int)
    default_stats = compute_business_cost(y_true, default_pred, amounts, cost_fp=cost_fp)

    savings_vs_baseline = baseline_unmitigated_cost - min_cost
    savings_vs_default = default_stats["total_cost"] - min_cost

    return {
        "model_name": model_name,
        "optimal_threshold": best_threshold,
        "min_cost": min_cost,
        "best_stats": best_stats,
        "default_threshold_cost": default_stats["total_cost"],
        "default_stats": default_stats,
        "baseline_unmitigated_cost": baseline_unmitigated_cost,
        "savings_vs_baseline": savings_vs_baseline,
        "savings_vs_default": savings_vs_default,
    }


# ==============================================================================
# EVALUATION RUNNER
# ==============================================================================


def run_evaluation(
    data_path: Path | None = None,
    preprocessed_path: Path | None = None,
    models_dir: Path = Path("models"),
    cost_fp: float = 10.0,
) -> dict[str, object]:
    """
    Main evaluation pipeline:
    1. Loads validation data and models.
    2. Computes Precision-Recall AUC (PR-AUC) and ROC-AUC.
    3. Runs threshold optimization loop minimizing financial business cost.
    4. Prints comparative cost-benefit results.
    """
    print("=" * 80)
    print("CREDIT CARD FRAUD EVALUATION & BUSINESS COST OPTIMIZATION")
    print("=" * 80)
    print(f"Cost Parameters:")
    print(f"  False Positive (FP) friction cost: ${cost_fp:.2f} per transaction")
    print(f"  False Negative (FN) fraud cost:    Actual transaction amount ($)")

    data = load_evaluation_data(data_path=data_path, preprocessed_path=preprocessed_path)
    X_val = data["X_val"]
    y_val = data["y_val"]
    amounts_val = data["amounts_val"]

    total_val_transactions = len(y_val)
    total_val_frauds = int(y_val.sum())
    total_fraud_volume = float(amounts_val[y_val == 1].sum())

    print(f"\nValidation Set Profile (Chronological Later Window):")
    print(f"  Total Transactions:   {total_val_transactions:,}")
    print(f"  Actual Fraud Count:   {total_val_frauds:,} ({(total_val_frauds/total_val_transactions)*100:.4f}%)")
    print(f"  Total Fraud Volume:   ${total_fraud_volume:,.2f}")

    results = {}

    # --- 1. Supervised XGBoost Evaluation ---
    xgb_path = models_dir / "xgboost_baseline.joblib"
    if not xgb_path.exists():
        xgb_path = models_dir / "xgboost_baseline.json"

    if xgb_path.exists():
        print("\n" + "=" * 80)
        print("MODEL 1: Supervised XGBoost Evaluation")
        print("=" * 80)

        if str(xgb_path).endswith(".joblib"):
            xgb_model = joblib.load(xgb_path)
            xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
        else:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)
            xgb_probs = xgb_model.predict_proba(X_val)[:, 1]

        xgb_pr_auc = average_precision_score(y_val, xgb_probs)
        xgb_roc_auc = roc_auc_score(y_val, xgb_probs)

        print(f"  Precision-Recall AUC (PR-AUC): {xgb_pr_auc:.4f}")
        print(f"  ROC-AUC:                       {xgb_roc_auc:.4f}")

        xgb_opt = find_optimal_threshold(
            y_true=y_val,
            y_scores=xgb_probs,
            amounts=amounts_val,
            model_name="Supervised XGBoost",
            cost_fp=cost_fp,
        )

        results["xgboost"] = {
            "pr_auc": float(xgb_pr_auc),
            "roc_auc": float(xgb_roc_auc),
            "optimization": xgb_opt,
        }
    else:
        print(f"\n[Warning] XGBoost model artifact not found in {models_dir}")

    # --- 2. Unsupervised Isolation Forest Evaluation ---
    iso_path = models_dir / "isolation_forest.joblib"
    if iso_path.exists():
        print("\n" + "=" * 80)
        print("MODEL 2: Unsupervised Isolation Forest Evaluation")
        print("=" * 80)

        iso_model = joblib.load(iso_path)
        # Raw decision function: negative for outliers, positive for inliers
        # Inverse yields higher score for fraud
        raw_anomaly = -iso_model.decision_function(X_val)

        # Min-max scale to [0, 1] for threshold sweep
        score_min = raw_anomaly.min()
        score_max = raw_anomaly.max()
        iso_scores = (raw_anomaly - score_min) / (score_max - score_min + 1e-12)

        iso_pr_auc = average_precision_score(y_val, iso_scores)
        iso_roc_auc = roc_auc_score(y_val, iso_scores)

        print(f"  Precision-Recall AUC (PR-AUC): {iso_pr_auc:.4f}")
        print(f"  ROC-AUC:                       {iso_roc_auc:.4f}")

        iso_opt = find_optimal_threshold(
            y_true=y_val,
            y_scores=iso_scores,
            amounts=amounts_val,
            model_name="Unsupervised Isolation Forest",
            cost_fp=cost_fp,
        )

        results["isolation_forest"] = {
            "pr_auc": float(iso_pr_auc),
            "roc_auc": float(iso_roc_auc),
            "optimization": iso_opt,
        }
    else:
        print(f"\n[Warning] Isolation Forest model artifact not found in {models_dir}")

    # --- 3. Summary & Financial Cost-Benefit Comparison ---
    print("\n" + "=" * 80)
    print("FINANCIAL COST-BENEFIT SUMMARY & OPTIMAL THRESHOLDS")
    print("=" * 80)
    print(f"Baseline Unmitigated Fraud Loss (No Model): ${total_fraud_volume:,.2f}\n")

    for m_key, m_data in results.items():
        opt = m_data["optimization"]
        stats = opt["best_stats"]
        def_stats = opt["default_stats"]

        print(f"Model: {opt['model_name']}")
        print(f"  PR-AUC:                   {m_data['pr_auc']:.4f}")
        print(f"  Optimal Threshold Cutoff: {opt['optimal_threshold']:.4f}")
        print(f"  Total Business Cost:      ${opt['min_cost']:,.2f}")
        print(f"    - FP Friction Cost:     ${stats['fp_cost']:,.2f} ({stats['fp_count']:,} false alarms @ ${cost_fp:.2f})")
        print(f"    - FN Fraud Loss:        ${stats['fn_cost']:,.2f} ({stats['fn_count']:,} missed frauds)")
        print(f"    - Intercepted Fraud:    ${stats['prevented_loss']:,.2f} ({stats['tp_count']:,} stopped frauds)")
        print(f"  Default Cutoff (0.5) Cost: ${opt['default_threshold_cost']:,.2f}")
        print(f"  Savings vs Default 0.5:   ${opt['savings_vs_default']:,.2f}")
        print(f"  Net Savings vs Baseline:  ${opt['savings_vs_baseline']:,.2f}")
        print("-" * 80)

    # Save summary artifact
    output_summary_path = models_dir / "evaluation_results.json"
    serializable_results = {
        k: {
            "pr_auc": v["pr_auc"],
            "roc_auc": v["roc_auc"],
            "optimal_threshold": v["optimization"]["optimal_threshold"],
            "min_total_cost": v["optimization"]["min_cost"],
            "fp_friction_cost": v["optimization"]["best_stats"]["fp_cost"],
            "fn_fraud_loss": v["optimization"]["best_stats"]["fn_cost"],
            "prevented_loss": v["optimization"]["best_stats"]["prevented_loss"],
            "savings_vs_baseline": v["optimization"]["savings_vs_baseline"],
            "savings_vs_default": v["optimization"]["savings_vs_default"],
        }
        for k, v in results.items()
    }

    with open(output_summary_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nSaved evaluation summary report to: {output_summary_path}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Fraud Models with PR-AUC and Business Cost Function Optimization"
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
        help="Directory containing trained model artifacts (default: models)",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=10.0,
        help="Cost ($) incurred per false positive friction event (default: $10.00)",
    )
    args = parser.parse_args()

    run_evaluation(
        data_path=Path(args.data_path) if args.data_path else None,
        preprocessed_path=Path(args.preprocessed_path) if args.preprocessed_path else None,
        models_dir=Path(args.models_dir),
        cost_fp=args.cost_fp,
    )


if __name__ == "__main__":
    main()
