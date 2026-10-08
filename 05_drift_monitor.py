#!/usr/bin/env python3
"""
05_drift_monitor.py - Kolmogorov-Smirnov (KS) Feature Drift Detection & Retraining Alerts
Monitors covariate and concept drift between baseline data/train_window.csv and data/later_window.csv.
Performs:
  1. Two-sample KS test across all numeric features (V1 through V28 and Amount).
  2. Model prediction confidence distribution comparison using saved XGBoost model.
  3. Automated alerting: triggers retraining alert if KS-test p-value < 0.05 for > 3 critical features
     or if prediction confidence shifts significantly.
  4. Formatted summary table of drifted features, KS statistics, and p-values.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib


# Critical fraud features identified by feature importance & correlation with transaction fraud
DEFAULT_CRITICAL_FEATURES = [
    "V14",
    "V17",
    "V12",
    "V10",
    "V4",
    "V11",
    "V2",
    "Amount",
]


def load_drift_datasets(
    train_path: Path = Path("data/train_window.csv"),
    later_path: Path = Path("data/later_window.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads baseline training window and simulated later operational window datasets."""
    if not train_path.exists():
        print(f"Error: Baseline dataset not found at {train_path}", file=sys.stderr)
        print("Please run 01_eda.py to generate dataset splits.", file=sys.stderr)
        sys.exit(1)

    if not later_path.exists():
        print(f"Error: Later window dataset not found at {later_path}", file=sys.stderr)
        print("Please run 01_eda.py to generate dataset splits.", file=sys.stderr)
        sys.exit(1)

    print("=" * 85)
    print(f"Loading baseline dataset from:     {train_path}")
    print(f"Loading operational dataset from:  {later_path}")
    print("=" * 85)

    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)

    print(f"Baseline Window Size:    {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    print(f"Operational Window Size: {later_df.shape[0]:,} rows, {later_df.shape[1]} columns")

    return train_df, later_df


# ==============================================================================
# 1. FEATURE DRIFT DETECTION VIA TWO-SAMPLE KS-TEST
# ==============================================================================


def detect_feature_drift(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    critical_features: list[str],
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Executes two-sample Kolmogorov-Smirnov test across all numeric features
    (e.g., V1 through V28 and Amount) to identify temporal distribution drift.
    """
    # Select numeric feature columns (excluding metadata / targets)
    exclude_cols = {"Class", "Time"}
    numeric_features = [
        col for col in train_df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    records = []

    for feature in numeric_features:
        train_vals = train_df[feature].dropna().values
        later_vals = later_df[feature].dropna().values

        ks_stat, p_val = ks_2samp(train_vals, later_vals)
        is_critical = feature in critical_features
        is_drifted = p_val < p_value_threshold

        # Severity classification based on KS statistic magnitude
        if ks_stat >= 0.10:
            severity = "HIGH"
        elif ks_stat >= 0.05:
            severity = "MODERATE"
        elif is_drifted:
            severity = "MILD"
        else:
            severity = "NONE"

        records.append({
            "feature": feature,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "is_critical": is_critical,
            "is_drifted": is_drifted,
            "severity": severity,
            "baseline_mean": float(np.mean(train_vals)),
            "later_mean": float(np.mean(later_vals)),
            "baseline_std": float(np.std(train_vals)),
            "later_std": float(np.std(later_vals)),
        })

    drift_df = pd.DataFrame(records)
    # Sort by KS statistic descending
    drift_df = drift_df.sort_values(by="ks_statistic", ascending=False).reset_index(drop=True)
    return drift_df


# ==============================================================================
# 2. MODEL PREDICTION CONFIDENCE DISTRIBUTION COMPARISON
# ==============================================================================


def compare_prediction_confidence(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    models_dir: Path = Path("models"),
    p_threshold: float = 0.05,
    ks_threshold: float = 0.02,
) -> dict[str, object] | None:
    """
    Evaluates shift in model prediction confidence distributions between the baseline
    and operational windows using the saved XGBoost classifier.
    """
    xgb_path = models_dir / "xgboost_baseline.joblib"
    if not xgb_path.exists():
        xgb_path = models_dir / "xgboost_baseline.json"

    if not xgb_path.exists():
        print(f"\n[Warning] XGBoost model artifact not found in {models_dir}. Skipping confidence comparison.")
        return None

    print(f"\nLoading XGBoost model from: {xgb_path}")
    if str(xgb_path).endswith(".joblib"):
        model = joblib.load(xgb_path)
    else:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(xgb_path)

    # Prepare features: scale Amount and Time using baseline fit (same as training pipeline)
    X_train = train_df.drop(columns=["Class"], errors="ignore").copy()
    X_later = later_df.drop(columns=["Class"], errors="ignore").copy()

    if "Amount" in X_train.columns:
        amt_scaler = RobustScaler().fit(X_train[["Amount"]])
        X_train["Amount"] = amt_scaler.transform(X_train[["Amount"]])
        X_later["Amount"] = amt_scaler.transform(X_later[["Amount"]])

    if "Time" in X_train.columns:
        time_scaler = StandardScaler().fit(X_train[["Time"]])
        X_train["Time"] = time_scaler.transform(X_train[["Time"]])
        X_later["Time"] = time_scaler.transform(X_later[["Time"]])

    print("Generating model prediction probabilities across both windows...")
    train_conf = model.predict_proba(X_train.values)[:, 1]
    later_conf = model.predict_proba(X_later.values)[:, 1]

    # Two-sample KS-test on prediction confidence distributions
    ks_stat, p_val = ks_2samp(train_conf, later_conf)
    significant_shift = (p_val < p_threshold) and (ks_stat >= ks_threshold)

    conf_summary = {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_val),
        "significant_shift": bool(significant_shift),
        "baseline_mean_confidence": float(np.mean(train_conf)),
        "later_mean_confidence": float(np.mean(later_conf)),
        "baseline_median_confidence": float(np.median(train_conf)),
        "later_median_confidence": float(np.median(later_conf)),
        "baseline_std_confidence": float(np.std(train_conf)),
        "later_std_confidence": float(np.std(later_conf)),
        "baseline_p95_confidence": float(np.percentile(train_conf, 95)),
        "later_p95_confidence": float(np.percentile(later_conf, 95)),
    }
    return conf_summary


# ==============================================================================
# 3. REPORTING, SUMMARY TABLE & RETRAINING ALERTS
# ==============================================================================


def evaluate_and_alert(
    drift_df: pd.DataFrame,
    conf_summary: dict[str, object] | None,
    critical_features: list[str],
    p_value_threshold: float = 0.05,
    max_tolerated_critical_drift: int = 3,
) -> dict[str, object]:
    """
    Prints summary tables and triggers automated retraining alert if:
    - KS-test p-value < 0.05 for > 3 critical features, OR
    - Prediction confidence distribution shifts significantly.
    """
    # 1. Feature Drift Summary Table
    print("\n" + "=" * 85)
    print("FEATURE DISTRIBUTION DRIFT SUMMARY TABLE (TWO-SAMPLE KS-TEST)")
    print("=" * 85)
    table_headers = f"{'Feature':<10} | {'KS Statistic':<14} | {'p-value':<14} | {'Critical?':<11} | {'Drift Status':<12}"
    print(table_headers)
    print("-" * len(table_headers))

    for _, row in drift_df.iterrows():
        crit_str = "YES (★)" if row["is_critical"] else "no"
        drift_str = f"DRIFT ({row['severity']})" if row["is_drifted"] else "STABLE"
        print(
            f"{row['feature']:<10} | "
            f"{row['ks_statistic']:<14.4f} | "
            f"{row['p_value']:<14.4e} | "
            f"{crit_str:<11} | "
            f"{drift_str:<12}"
        )

    print("-" * len(table_headers))

    # Critical features breakdown
    crit_df = drift_df[drift_df["is_critical"]]
    drifted_crit_df = crit_df[crit_df["is_drifted"]]
    num_drifted_critical = len(drifted_crit_df)

    print(f"\n[Feature Drift Counts]")
    print(f"  Total Numerical Features Checked:       {len(drift_df)}")
    print(f"  Total Features with Drift (p < {p_value_threshold}):   {drift_df['is_drifted'].sum()}")
    print(f"  Critical Features Monitored:            {len(crit_df)}")
    print(f"  Critical Features with Drift:           {num_drifted_critical} of {len(crit_df)}")
    print(f"  Critical Feature Drift Allowed:         <= {max_tolerated_critical_drift}")

    # 2. Prediction Confidence Distribution Table
    conf_shifted = False
    if conf_summary:
        print("\n" + "=" * 85)
        print("MODEL PREDICTION CONFIDENCE DISTRIBUTION COMPARISON (XGBOOST)")
        print("=" * 85)
        print(f"Confidence KS Statistic:    {conf_summary['ks_statistic']:.4f}")
        print(f"Confidence KS p-value:      {conf_summary['p_value']:.4e}")
        print(f"Significant Shift Detected: {'YES' if conf_summary['significant_shift'] else 'NO'}\n")

        print(f"{'Metric':<28} | {'Baseline Window':<20} | {'Operational Window':<20}")
        print("-" * 74)
        print(f"{'Mean Fraud Confidence':<28} | {conf_summary['baseline_mean_confidence']:<20.6f} | {conf_summary['later_mean_confidence']:<20.6f}")
        print(f"{'Median Fraud Confidence':<28} | {conf_summary['baseline_median_confidence']:<20.6f} | {conf_summary['later_median_confidence']:<20.6f}")
        print(f"{'Std Fraud Confidence':<28} | {conf_summary['baseline_std_confidence']:<20.6f} | {conf_summary['later_std_confidence']:<20.6f}")
        print(f"{'95th Percentile Confidence':<28} | {conf_summary['baseline_p95_confidence']:<20.6f} | {conf_summary['later_p95_confidence']:<20.6f}")
        print("-" * 74)

        conf_shifted = conf_summary["significant_shift"]

    # 3. Alert Decision Logic
    critical_feature_trigger = num_drifted_critical > max_tolerated_critical_drift
    retraining_alert = critical_feature_trigger or conf_shifted

    print("\n" + "=" * 85)
    if retraining_alert:
        drifted_names = ", ".join(drifted_crit_df["feature"].tolist()) if num_drifted_critical > 0 else "None"
        print("🚨" * 35)
        print("⚠️  [PROMINENT WARNING: AUTOMATIC MODEL RETRAINING FLAGGED]  ⚠️")
        print("🚨" * 35)
        print("\nRoot Causes:")
        if critical_feature_trigger:
            print(f"  • CRITICAL FEATURE DRIFT: {num_drifted_critical} critical features exceeded drift threshold (p < {p_value_threshold}).")
            print(f"    Affected features: {drifted_names}")
        if conf_shifted:
            print(f"  • PREDICTION CONFIDENCE SHIFT: Model confidence distribution shifted significantly")
            print(f"    (KS stat = {conf_summary['ks_statistic']:.4f}, p = {conf_summary['p_value']:.4e}).")

        print("\nImpact on Production System:")
        print("  - Statistical shift between training baseline and operational stream invalidates model calibration.")
        print("  - Fraud capture recall may erode and customer friction rates will fluctuate.")

        print("\nAutomated Action:")
        print("  >>> System flagged for AUTOMATIC RETRAINING on latest operational data.")
        print("=" * 85)
    else:
        print("✅" * 35)
        print("   [SYSTEM STATUS: STABLE - NO RETRAINING REQUIRED]   ")
        print("✅" * 35)
        print(f"\nFeature distributions and confidence outputs remain within acceptable bounds.")
        print("=" * 85)

    alert_result = {
        "retraining_alert": bool(retraining_alert),
        "critical_feature_trigger": bool(critical_feature_trigger),
        "confidence_shift_trigger": bool(conf_shifted),
        "drifted_critical_count": int(num_drifted_critical),
        "drifted_critical_features": drifted_crit_df["feature"].tolist(),
        "total_drifted_features": int(drift_df["is_drifted"].sum()),
        "confidence_summary": conf_summary,
    }
    return alert_result


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Kolmogorov-Smirnov Feature & Prediction Confidence Drift"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/train_window.csv",
        help="Path to baseline data/train_window.csv (default: data/train_window.csv)",
    )
    parser.add_argument(
        "--later-path",
        type=str,
        default="data/later_window.csv",
        help="Path to simulated operational data/later_window.csv (default: data/later_window.csv)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory with trained model artifacts (default: models)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="p-value significance threshold for KS-test drift (default: 0.05)",
    )
    parser.add_argument(
        "--max-critical-drift",
        type=int,
        default=3,
        help="Maximum tolerated critical features with drift before alert (default: 3)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="models/drift_report.json",
        help="Path to save JSON monitoring report (default: models/drift_report.json)",
    )
    args = parser.parse_args()

    train_df, later_df = load_drift_datasets(
        train_path=Path(args.train_path),
        later_path=Path(args.later_path),
    )

    drift_df = detect_feature_drift(
        train_df=train_df,
        later_df=later_df,
        critical_features=DEFAULT_CRITICAL_FEATURES,
        p_value_threshold=args.p_threshold,
    )

    conf_summary = compare_prediction_confidence(
        train_df=train_df,
        later_df=later_df,
        models_dir=Path(args.models_dir),
        p_threshold=args.p_threshold,
    )

    alert_result = evaluate_and_alert(
        drift_df=drift_df,
        conf_summary=conf_summary,
        critical_features=DEFAULT_CRITICAL_FEATURES,
        p_value_threshold=args.p_threshold,
        max_tolerated_critical_drift=args.max_critical_drift,
    )

    # Save output report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "alert_summary": alert_result,
        "feature_drift_metrics": drift_df.to_dict(orient="records"),
    }
    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved drift monitoring report to: {report_path}\n")


if __name__ == "__main__":
    main()
