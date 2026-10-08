#!/usr/bin/env python3
"""
04_evaluate.py - Precision-Recall AUC Evaluation & Cost-Benefit Threshold Optimization
Evaluates Supervised XGBoost and Unsupervised Isolation Forest on validation data.
Implements custom business cost function:
  - False Positive (FP): $10 in customer friction / investigation cost
  - False Negative (FN): Actual transaction Amount in unrecovered fraud losses
Sweeps decision thresholds (0.01 to 0.99) to find the cutoff that minimizes total business cost.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
import joblib


# ==============================================================================
# INLINE COMMENTARY: WHY PR-AUC IS PRIORITIZED OVER ROC-AUC FOR EXTREME FRAUD IMBALANCE
# ==============================================================================
#
# In extreme class imbalance settings (e.g., credit card fraud where the minority
# fraud class constitutes only ~0.17% of total transactions), ROC-AUC is fundamentally
# poorly adapted and becomes falsely optimistic due to the massive volume of True Negatives (TN).
#
# Mathematical Reason:
# - ROC Curve evaluates False Positive Rate (FPR) vs. True Positive Rate (TPR / Recall):
#       FPR = FP / (FP + TN)
#   Because TN is colossal (tens or hundreds of thousands of legitimate transactions),
#   the denominator (FP + TN) is overwhelming. Even if the model generates hundreds of
#   false alarms (high FP), FPR remains tiny (e.g., FPR < 0.005).
#   Consequently, the ROC curve appears nearly ideal (ROC-AUC > 0.95), masking disastrous
#   precision degradation in production.
#
# - PR Curve evaluates Precision vs. Recall:
#       Precision = TP / (TP + FP)
#       Recall    = TP / (TP + FN)
#   Precision directly pits False Positives against True Positives, completely
#   unaffected by the massive number of True Negatives. If the model flags 100 transactions
#   to catch 5 frauds (95 false alarms), Precision plunges to 5%, which PR-AUC
#   immediately exposes.
#
# Therefore, Precision-Recall AUC (PR-AUC) is prioritized as the primary metric for
# model selection and performance evaluation in financial fraud detection.
# ==============================================================================


def load_validation_data(
    val_path: Path | None = None,
    train_window_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Loads validation features, ground truth labels, and unscaled transaction amounts.
    Returns:
        (X_val, y_val, amounts_val, feature_names)
    """
    val_csv = val_path if val_path else Path("data/val_preprocessed.csv")
    tw_csv = train_window_path if train_window_path else Path("data/train_window.csv")

    if val_csv.exists():
        print(f"Loading preprocessed validation split from: {val_csv}")
        val_df = pd.read_csv(val_csv)

        if "Class" not in val_df.columns:
            print("Error: 'Class' column missing from validation dataset.", file=sys.stderr)
            sys.exit(1)

        X_val_df = val_df.drop(columns=["Class"])
        y_val = val_df["Class"].astype(int).values
        feature_names = list(X_val_df.columns)
        X_val = X_val_df.values

        # Retrieve exact unscaled transaction amounts for cost function
        amounts_val = None
        if tw_csv.exists():
            tw_df = pd.read_csv(tw_csv)
            if "Time" in tw_df.columns:
                tw_df = tw_df.sort_values(by="Time").reset_index(drop=True)
            if len(tw_df) >= len(val_df):
                # Validation split corresponds to the chronological tail
                amounts_val = tw_df.iloc[-len(val_df):]["Amount"].values.astype(float)

        if amounts_val is None or len(amounts_val) != len(y_val):
            # Fallback to Amount column if unscaled or positive approximation
            if "Amount" in val_df.columns:
                amounts_val = np.maximum(val_df["Amount"].values.astype(float), 1.0)
            else:
                amounts_val = np.full(len(y_val), 88.0)

        return X_val, y_val, amounts_val, feature_names

    # Check compressed npz archive
    npz_candidates = [
        Path("data/preprocessed_data.npz"),
        Path("data/processed/preprocessed_data.npz"),
    ]
    for npz_path in npz_candidates:
        if npz_path.exists():
            print(f"Loading validation split from archive: {npz_path}")
            npz = np.load(npz_path, allow_pickle=True)
            X_val = npz["X_later_scaled"]
            y_val = npz["y_later"]
            amounts_val = np.full(len(y_val), 88.0)  # average amount fallback
            feature_names = [f"feat_{i}" for i in range(X_val.shape[1])]
            return X_val, y_val, amounts_val, feature_names

    print(
        "Error: Could not locate validation dataset. Please run 01_eda.py and 02_preprocess.py first.",
        file=sys.stderr,
    )
    sys.exit(1)


# ==============================================================================
# BUSINESS COST FUNCTION & THRESHOLD OPTIMIZATION
# ==============================================================================


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cost_fp: float = 10.0,
) -> dict[str, float | int]:
    """
    Calculates total business financial impact:
    - False Positive (FP): $10 friction cost (customer friction + review)
    - False Negative (FN): Loss equal to actual fraudulent transaction Amount
    - True Positive (TP): Fraud intercepted ($0 fraud loss)
    - True Negative (TN): Genuine transaction approved ($0 friction cost)
    """
    is_fraud = (y_true == 1)
    is_pred_fraud = (y_pred == 1)

    # FP: Predicted fraud (1) but actually legitimate (0)
    fp_mask = (~is_fraud) & is_pred_fraud
    fp_cost = float(fp_mask.sum() * cost_fp)

    # FN: Actual fraud (1) missed by model (0) -> actual unrecovered fraud amount
    fn_mask = is_fraud & (~is_pred_fraud)
    fn_cost = float(amounts[fn_mask].sum())

    # TP: Intercepted fraud volume
    tp_mask = is_fraud & is_pred_fraud
    prevented_loss = float(amounts[tp_mask].sum())

    total_cost = fp_cost + fn_cost

    return {
        "total_cost": total_cost,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
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
    threshold_range: np.ndarray = np.arange(0.01, 1.00, 0.01),
    cost_fp: float = 10.0,
) -> dict[str, object]:
    """
    Iterates through candidate decision thresholds (from 0.01 to 0.99)
    to find the specific cutoff that strictly minimizes total financial business cost.
    """
    baseline_unmitigated_cost = float(amounts[y_true == 1].sum())

    best_threshold = 0.50
    min_cost = float("inf")
    best_stats = None

    threshold_results = []

    # Threshold optimization loop
    for threshold in threshold_range:
        y_pred = (y_scores >= threshold).astype(int)
        stats = compute_business_cost(y_true, y_pred, amounts, cost_fp=cost_fp)
        cost = stats["total_cost"]

        threshold_results.append({
            "threshold": round(float(threshold), 4),
            "total_cost": cost,
            "fp_cost": stats["fp_cost"],
            "fn_cost": stats["fn_cost"],
            "fp_count": stats["fp_count"],
            "fn_count": stats["fn_count"],
            "tp_count": stats["tp_count"],
        })

        if cost < min_cost:
            min_cost = cost
            best_threshold = round(float(threshold), 4)
            best_stats = stats

    # Evaluate default 0.50 cutoff for cost-benefit comparison
    default_pred = (y_scores >= 0.50).astype(int)
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
        "threshold_curve": threshold_results,
    }


# ==============================================================================
# MAIN EVALUATION RUNNER
# ==============================================================================


def run_evaluation(
    models_dir: Path = Path("models"),
    val_path: Path | None = None,
    cost_fp: float = 10.0,
) -> dict[str, object]:
    """
    Main evaluation routine:
    1. Loads validation split and trained models.
    2. Calculates PR-AUC and ROC-AUC.
    3. Runs threshold optimization loop across [0.01, 0.99].
    4. Prints evaluation summary and optimal threshold cutoffs.
    """
    print("=" * 80)
    print("EVALUATION PIPELINE: PR-AUC & BUSINESS COST THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Business Cost Function Parameters:")
    print(f"  False Positive (FP) friction cost: ${cost_fp:.2f} per transaction")
    print(f"  False Negative (FN) fraud loss:    Actual transaction Amount ($)")

    X_val, y_val, amounts_val, feature_names = load_validation_data(val_path=val_path)

    total_samples = len(y_val)
    total_frauds = int(y_val.sum())
    total_fraud_dollars = float(amounts_val[y_val == 1].sum())

    print(f"\nValidation Set Statistics:")
    print(f"  Total Transactions: {total_samples:,}")
    print(f"  Fraud Transactions: {total_frauds:,} ({total_frauds / total_samples:.4%})")
    print(f"  Total Fraud Volume: ${total_fraud_dollars:,.2f}")
    print(f"  Unmitigated Baseline Fraud Loss: ${total_fraud_dollars:,.2f}")

    results = {}
    threshold_range = np.arange(0.01, 1.00, 0.01)

    # --------------------------------------------------------------------------
    # 1. Supervised XGBoost Evaluation
    # --------------------------------------------------------------------------
    xgb_path = models_dir / "xgboost_baseline.joblib"
    if not xgb_path.exists():
        xgb_path = models_dir / "xgboost_baseline.json"

    if xgb_path.exists():
        print("\n" + "=" * 80)
        print("MODEL 1: Supervised XGBoost Classifier")
        print("=" * 80)

        if str(xgb_path).endswith(".joblib"):
            xgb_model = joblib.load(xgb_path)
            xgb_scores = xgb_model.predict_proba(X_val)[:, 1]
        else:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)
            xgb_scores = xgb_model.predict_proba(X_val)[:, 1]

        xgb_pr_auc = average_precision_score(y_val, xgb_scores)
        xgb_roc_auc = roc_auc_score(y_val, xgb_scores)

        print(f"  PR-AUC Score (Prioritized Metric): {xgb_pr_auc:.4f}")
        print(f"  ROC-AUC Score (Reference Metric):   {xgb_roc_auc:.4f}")

        xgb_opt = find_optimal_threshold(
            y_true=y_val,
            y_scores=xgb_scores,
            amounts=amounts_val,
            model_name="Supervised XGBoost",
            threshold_range=threshold_range,
            cost_fp=cost_fp,
        )

        results["xgboost"] = {
            "pr_auc": float(xgb_pr_auc),
            "roc_auc": float(xgb_roc_auc),
            "optimization": xgb_opt,
        }

        print(f"\n  [Threshold Optimization Results]:")
        print(f"    Chosen Optimal Threshold: {xgb_opt['optimal_threshold']:.2f}")
        print(f"    Min Total Business Cost:  ${xgb_opt['min_cost']:,.2f}")
        print(f"      - FP Friction Cost:     ${xgb_opt['best_stats']['fp_cost']:,.2f} ({xgb_opt['best_stats']['fp_count']} false alerts @ ${cost_fp:.2f})")
        print(f"      - FN Fraud Loss:        ${xgb_opt['best_stats']['fn_cost']:,.2f} ({xgb_opt['best_stats']['fn_count']} missed frauds)")
        print(f"      - Recovered Fraud Loss: ${xgb_opt['best_stats']['prevented_loss']:,.2f} ({xgb_opt['best_stats']['tp_count']} caught frauds)")
        print(f"    Cost at Default (0.50):   ${xgb_opt['default_threshold_cost']:,.2f}")
        print(f"    Financial Savings vs 0.5: ${xgb_opt['savings_vs_default']:,.2f}")
    else:
        print(f"\n[Warning] XGBoost model artifact not found in {models_dir}")

    # --------------------------------------------------------------------------
    # 2. Unsupervised Isolation Forest Evaluation
    # --------------------------------------------------------------------------
    iso_path = models_dir / "isolation_forest.joblib"
    if iso_path.exists():
        print("\n" + "=" * 80)
        print("MODEL 2: Unsupervised Isolation Forest")
        print("=" * 80)

        iso_model = joblib.load(iso_path)
        # decision_function gives negative values for anomalies; inverting gives higher score for fraud
        raw_anomaly = -iso_model.decision_function(X_val)

        # Min-max scale anomaly score to [0, 1] so [0.01, 0.99] threshold sweep applies directly
        score_min = raw_anomaly.min()
        score_max = raw_anomaly.max()
        iso_scores = (raw_anomaly - score_min) / (score_max - score_min + 1e-12)

        iso_pr_auc = average_precision_score(y_val, iso_scores)
        iso_roc_auc = roc_auc_score(y_val, iso_scores)

        print(f"  PR-AUC Score (Prioritized Metric): {iso_pr_auc:.4f}")
        print(f"  ROC-AUC Score (Reference Metric):   {iso_roc_auc:.4f}")

        iso_opt = find_optimal_threshold(
            y_true=y_val,
            y_scores=iso_scores,
            amounts=amounts_val,
            model_name="Unsupervised Isolation Forest",
            threshold_range=threshold_range,
            cost_fp=cost_fp,
        )

        results["isolation_forest"] = {
            "pr_auc": float(iso_pr_auc),
            "roc_auc": float(iso_roc_auc),
            "optimization": iso_opt,
        }

        print(f"\n  [Threshold Optimization Results]:")
        print(f"    Chosen Optimal Threshold: {iso_opt['optimal_threshold']:.2f}")
        print(f"    Min Total Business Cost:  ${iso_opt['min_cost']:,.2f}")
        print(f"      - FP Friction Cost:     ${iso_opt['best_stats']['fp_cost']:,.2f} ({iso_opt['best_stats']['fp_count']} false alerts @ ${cost_fp:.2f})")
        print(f"      - FN Fraud Loss:        ${iso_opt['best_stats']['fn_cost']:,.2f} ({iso_opt['best_stats']['fn_count']} missed frauds)")
        print(f"      - Recovered Fraud Loss: ${iso_opt['best_stats']['prevented_loss']:,.2f} ({iso_opt['best_stats']['tp_count']} caught frauds)")
        print(f"    Cost at Default (0.50):   ${iso_opt['default_threshold_cost']:,.2f}")
        print(f"    Financial Savings vs 0.5: ${iso_opt['savings_vs_default']:,.2f}")
    else:
        print(f"\n[Warning] Isolation Forest model artifact not found in {models_dir}")

    # --------------------------------------------------------------------------
    # 3. Comparative Summary Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY: PR-AUC SCORES & OPTIMAL THRESHOLDS")
    print("=" * 80)
    summary_headers = f"{'Model':<30} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Threshold':<10} | {'Total Cost ($)':<14}"
    print(summary_headers)
    print("-" * len(summary_headers))

    for m_key, m_data in results.items():
        opt = m_data["optimization"]
        print(
            f"{opt['model_name']:<30} | "
            f"{m_data['pr_auc']:<8.4f} | "
            f"{m_data['roc_auc']:<8.4f} | "
            f"{opt['optimal_threshold']:<10.2f} | "
            f"${opt['min_cost']:<13,.2f}"
        )

    print("=" * 80)

    # Save evaluation summary to models/evaluation_results.json
    output_summary_path = models_dir / "evaluation_results.json"
    serializable = {
        m_key: {
            "model_name": m_data["optimization"]["model_name"],
            "pr_auc": round(m_data["pr_auc"], 4),
            "roc_auc": round(m_data["roc_auc"], 4),
            "optimal_threshold": m_data["optimization"]["optimal_threshold"],
            "min_total_cost": round(m_data["optimization"]["min_cost"], 2),
            "fp_cost": round(m_data["optimization"]["best_stats"]["fp_cost"], 2),
            "fn_cost": round(m_data["optimization"]["best_stats"]["fn_cost"], 2),
            "prevented_loss": round(m_data["optimization"]["best_stats"]["prevented_loss"], 2),
            "savings_vs_baseline": round(m_data["optimization"]["savings_vs_baseline"], 2),
            "savings_vs_default": round(m_data["optimization"]["savings_vs_default"], 2),
        }
        for m_key, m_data in results.items()
    }

    with open(output_summary_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved evaluation metrics to: {output_summary_path}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Fraud Models with PR-AUC and Business Cost Function Optimization"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained model artifacts (default: models)",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Path to validation CSV (defaults to data/val_preprocessed.csv)",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=10.0,
        help="Customer friction cost ($) per false positive alarm (default: $10.00)",
    )
    args = parser.parse_args()

    run_evaluation(
        models_dir=Path(args.models_dir),
        val_path=Path(args.val_path) if args.val_path else None,
        cost_fp=args.cost_fp,
    )


if __name__ == "__main__":
    main()
