#!/usr/bin/env python3
"""
05_drift_monitor.py - Kolmogorov-Smirnov (KS) Feature Drift Monitor & Retraining Alerts
Compares feature distributions between the baseline 'training_window' and simulated 'later_window'.
Computes KS-statistic drift scores to detect covariate/temporal distribution shift.
Triggers an automated retraining alert if drift score exceeds 0.05 for more than 3 critical features.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Dynamic module loading
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

# Default critical fraud features strongly correlated with transaction legitimacy
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


def run_ks_2samp(sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
    """
    Computes two-sample Kolmogorov-Smirnov statistic and p-value.
    Uses scipy.stats.ks_2samp if available, with pure NumPy fallback.
    """
    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(sample1, sample2)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        # Pure NumPy two-sample KS calculation fallback
        n1 = len(sample1)
        n2 = len(sample2)
        data1_sorted = np.sort(sample1)
        data2_sorted = np.sort(sample2)
        all_points = np.concatenate([data1_sorted, data2_sorted])
        cdf1 = np.searchsorted(data1_sorted, all_points, side="right") / n1
        cdf2 = np.searchsorted(data2_sorted, all_points, side="right") / n2
        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # Asymptotic p-value approximation
        en = np.sqrt(n1 * n2 / (n1 + n2))
        lambda_val = (en + 0.12 + 0.11 / en) * ks_stat
        if lambda_val <= 0:
            p_val = 1.0
        else:
            terms = [2 * ((-1) ** (j - 1)) * np.exp(-2 * (j**2) * (lambda_val**2)) for j in range(1, 101)]
            p_val = float(np.clip(np.sum(terms), 0.0, 1.0))
        return ks_stat, p_val


def load_drift_data(
    data_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads dataset and partitions chronologically into 'training_window' (first 70%)
    and 'later_window' (last 30%) to simulate production time drift.
    """
    raw_path = data_path if data_path else (preprocess_mod.find_dataset() if preprocess_mod else None)

    if raw_path and raw_path.exists():
        print(f"Loading dataset for drift monitoring from: {raw_path}")
        df = pd.read_csv(raw_path)
        training_window, later_window = preprocess_mod.chronological_split(df, train_ratio=0.70)
        return training_window, later_window

    raise FileNotFoundError(
        "Credit Card Fraud dataset not found. Please provide path via --data-path <path/to/creditcard.csv>."
    )


def compute_drift_metrics(
    training_window: pd.DataFrame,
    later_window: pd.DataFrame,
    critical_features: list[str],
    drift_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculates the two-sample Kolmogorov-Smirnov drift score across all numerical features.
    Flags whether drift score exceeds the critical cutoff (0.05).
    """
    feature_cols = [
        col for col in training_window.columns
        if col not in ["Class", "Time"] and pd.api.types.is_numeric_dtype(training_window[col])
    ]

    records = []

    for col in feature_cols:
        baseline_vals = training_window[col].dropna().values
        later_vals = later_window[col].dropna().values

        ks_stat, p_val = run_ks_2samp(baseline_vals, later_vals)
        is_critical = col in critical_features
        is_drifted = ks_stat > drift_threshold

        # Categorize drift severity
        if ks_stat >= 0.15:
            severity = "CRITICAL"
        elif ks_stat >= drift_threshold:
            severity = "MODERATE"
        else:
            severity = "NEGLIGIBLE"

        records.append({
            "feature": col,
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "is_critical": is_critical,
            "is_drifted": is_drifted,
            "severity": severity,
            "baseline_mean": float(baseline_vals.mean()),
            "later_mean": float(later_vals.mean()),
            "baseline_std": float(baseline_vals.std()),
            "later_std": float(later_vals.std()),
        })

    drift_df = pd.DataFrame(records).sort_values(by="ks_statistic", ascending=False).reset_index(drop=True)
    return drift_df


def evaluate_drift_and_alert(
    drift_df: pd.DataFrame,
    critical_features: list[str],
    drift_threshold: float = 0.05,
    max_tolerated_critical_drift: int = 3,
) -> dict[str, object]:
    """
    Evaluates drift across critical features and triggers a retraining alert
    if drift score crosses the threshold (> 0.05) for more than three critical features.
    """
    print("\n" + "=" * 90)
    print("KOLMOGOROV-SMIRNOV FEATURE DRIFT ANALYSIS REPORT")
    print("=" * 90)

    # Filter critical features
    critical_drift_df = drift_df[drift_df["is_critical"]].copy()
    drifted_critical = critical_drift_df[critical_drift_df["is_drifted"]]
    num_drifted_critical = len(drifted_critical)

    # Print summary table
    print(f"{'Feature':<12} | {'KS Drift Score':<15} | {'p-value':<12} | {'Critical?':<10} | {'Status':<12}")
    print("-" * 75)

    for _, row in drift_df.iterrows():
        crit_label = "YES (★)" if row["is_critical"] else "no"
        flag = f"DRIFT ({row['severity']})" if row["is_drifted"] else "STABLE"
        print(
            f"{row['feature']:<12} | {row['ks_statistic']:<15.4f} | {row['p_value']:<12.4e} | {crit_label:<10} | {flag:<12}"
        )

    print("-" * 75)
    print(f"\n[Summary]")
    print(f"  Total features analyzed:           {len(drift_df)}")
    print(f"  Total features with KS > {drift_threshold:.2f}:       {drift_df['is_drifted'].sum()}")
    print(f"  Critical features analyzed:        {len(critical_drift_df)}")
    print(f"  Critical features with KS > {drift_threshold:.2f}:     {num_drifted_critical}")
    print(f"  Tolerated critical drift limit:    {max_tolerated_critical_drift}")

    # Alert Decision Logic
    retraining_required = num_drifted_critical > max_tolerated_critical_drift

    print("\n" + "=" * 90)
    if retraining_required:
        drifted_names = ", ".join(drifted_critical["feature"].tolist())
        print("🚨 [ALERT: MODEL RETRAINING RECOMMENDED] 🚨")
        print("=" * 90)
        print(f"CRITICAL DRIFT DETECTED: {num_drifted_critical} critical features crossed the KS drift threshold ({drift_threshold:.2f}).")
        print(f"Affected critical features: {drifted_names}")
        print("\nReasoning:")
        print("  - The empirical distribution of critical predictive fraud signals has significantly drifted")
        print("    between the training baseline and the later operational window.")
        print("  - Continued inference without updating model weights may degrade fraud capture rates (Recall)")
        print("    and induce elevated false alarms (Friction costs).")
        print("\nRecommended Actions:")
        print("  1. Trigger automated retraining pipeline on recent chronological batches.")
        print("  2. Re-calibrate decision threshold (via 04_evaluate.py cost function).")
        print("  3. Log drift metrics to monitoring dashboard.")
    else:
        print("✅ [STATUS: FEATURE DISTRIBUTIONS STABLE - NO RETRAINING NEEDED] ✅")
        print("=" * 90)
        print(f"Only {num_drifted_critical} critical feature(s) showed KS drift score > {drift_threshold:.2f}.")
        print(f"This is within the acceptable threshold (<= {max_tolerated_critical_drift} critical features).")

    print("=" * 90)

    alert_summary = {
        "retraining_required": bool(retraining_required),
        "num_drifted_critical": int(num_drifted_critical),
        "max_tolerated": int(max_tolerated_critical_drift),
        "drift_threshold": float(drift_threshold),
        "drifted_critical_features": drifted_critical["feature"].tolist(),
        "total_drifted_features": int(drift_df["is_drifted"].sum()),
    }
    return alert_summary


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Kolmogorov-Smirnov Feature Drift and Trigger Retraining Alerts"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to creditcard.csv dataset",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="KS statistic cutoff threshold for drift flagging (default: 0.05)",
    )
    parser.add_argument(
        "--max-critical-drift",
        type=int,
        default=3,
        help="Maximum allowed critical features with drift before triggering alert (default: 3)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="models/drift_report.json",
        help="Path to save drift monitoring JSON report (default: models/drift_report.json)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else None

    try:
        training_window, later_window = load_drift_data(data_path=data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    drift_df = compute_drift_metrics(
        training_window=training_window,
        later_window=later_window,
        critical_features=DEFAULT_CRITICAL_FEATURES,
        drift_threshold=args.drift_threshold,
    )

    alert_summary = evaluate_drift_and_alert(
        drift_df=drift_df,
        critical_features=DEFAULT_CRITICAL_FEATURES,
        drift_threshold=args.drift_threshold,
        max_tolerated_critical_drift=args.max_critical_drift,
    )

    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "summary": alert_summary,
            "feature_metrics": drift_df.to_dict(orient="records"),
        }
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nSaved drift monitoring report to: {report_path}")


if __name__ == "__main__":
    main()
