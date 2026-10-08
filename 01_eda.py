#!/usr/bin/env python3
"""
01_eda.py - Exploratory Data Analysis & Class Imbalance Calculation
Kaggle Credit Card Fraud Detection Dataset
"""

import argparse
from pathlib import Path
import sys
import pandas as pd


def find_dataset(default_filename: str = "creditcard.csv") -> Path | None:
    """Search for the dataset in common project paths."""
    candidates = [
        Path(default_filename),
        Path("data") / default_filename,
        Path("..") / default_filename,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.70
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split the dataset into a 'training_window' (first 70%)
    and a 'later_window' (last 30%) to simulate time drift, ensuring data is not shuffled.
    """
    if "Time" in df.columns:
        df_sorted = df.sort_values(by="Time").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    split_idx = int(len(df_sorted) * train_ratio)
    training_window = df_sorted.iloc[:split_idx].copy()
    later_window = df_sorted.iloc[split_idx:].copy()

    return training_window, later_window


def run_eda(data_path: Path) -> None:
    """Load dataset, print basic statistics, and compute class imbalance."""
    print("=" * 80)
    print(f"Loading Kaggle Credit Card Fraud dataset from: {data_path}")
    print("=" * 80)

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- 1. Dataset Overview ---")
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory Usage:  {memory_mb:.2f} MB")

    print("\n--- 2. Missing Values Check ---")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"Total Missing Values across all columns: {total_missing}")
    if total_missing > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")

    print("\n--- 3. Basic Descriptive Statistics ---")
    key_cols = [col for col in ["Time", "Amount"] if col in df.columns]
    if key_cols:
        print("Summary statistics for key features:")
        print(df[key_cols].describe().to_string())

    if "Class" not in df.columns:
        print("\nError: 'Class' column not found in dataset.", file=sys.stderr)
        sys.exit(1)

    print("\n--- 4. Class Imbalance Analysis ---")
    class_counts = df["Class"].value_counts().sort_index()
    total_samples = len(df)

    count_legit = class_counts.get(0, 0)
    count_fraud = class_counts.get(1, 0)

    pct_legit = (count_legit / total_samples) * 100 if total_samples > 0 else 0
    pct_fraud = (count_fraud / total_samples) * 100 if total_samples > 0 else 0

    print(f"Total Transactions:        {total_samples:,}")
    print(f"Legitimate (Class 0):      {count_legit:,} ({pct_legit:.4f}%)")
    print(f"Fraudulent (Class 1):      {count_fraud:,} ({pct_fraud:.4f}%)")

    if count_fraud > 0:
        imbalance_ratio = count_legit / count_fraud
        print(f"\nExact Imbalance Ratio:     {imbalance_ratio:.2f} : 1  (Legitimate : Fraud)")
        print(f"Inverse Ratio (Fraud %):   {pct_fraud:.6f}%")
        print(f"Fraction (1 in N):         1 fraud in every ~{int(round(imbalance_ratio + 1))} transactions")
    else:
        print("\nWarning: No fraud samples (Class 1) found in the dataset.")

    if "Amount" in df.columns:
        print("\n--- 5. Transaction Amount Comparison by Class ---")
        amount_stats = df.groupby("Class")["Amount"].agg(["count", "mean", "std", "median", "min", "max"])
        amount_stats.index = ["Legitimate (0)", "Fraud (1)"] if len(amount_stats) == 2 else amount_stats.index
        print(amount_stats.to_string())

    print("\n--- 6. Chronological Split (Time Drift Simulation) ---")
    training_window, later_window = chronological_split(df)
    
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train_window.csv"
    later_path = data_dir / "later_window.csv"

    training_window.to_csv(train_path, index=False)
    later_window.to_csv(later_path, index=False)

    print(f"Training Window Shape: {training_window.shape}")
    print(f"Later Window Shape:    {later_window.shape}")

    if "Time" in df.columns:
        print(
            f"  Training Window Time: [{training_window['Time'].min():.1f}s, {training_window['Time'].max():.1f}s]"
        )
        print(
            f"  Later Window Time:    [{later_window['Time'].min():.1f}s, {later_window['Time'].max():.1f}s]"
        )
    if "Class" in df.columns:
        tr_fraud = int(training_window["Class"].sum())
        tr_pct = (tr_fraud / len(training_window)) * 100 if len(training_window) > 0 else 0
        lt_fraud = int(later_window["Class"].sum())
        lt_pct = (lt_fraud / len(later_window)) * 100 if len(later_window) > 0 else 0
        print(f"Training Window Fraud Count: {tr_fraud:,} ({tr_pct:.4f}%)")
        print(f"Later Window Fraud Count:    {lt_fraud:,} ({lt_pct:.4f}%)")

    print(f"\nSaved CSV splits:")
    print(f"  - {train_path}")
    print(f"  - {later_path}")

    print("\n" + "=" * 80)
    print("EDA Complete.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="EDA and Class Imbalance for Credit Card Fraud Dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to creditcard.csv dataset (defaults to auto-detecting creditcard.csv or data/creditcard.csv)",
    )
    args = parser.parse_args()

    if args.data_path:
        target_path = Path(args.data_path)
    else:
        target_path = find_dataset()

    if target_path is None or not target_path.exists():
        print("Dataset not found!", file=sys.stderr)
        print("Please place 'creditcard.csv' in the project directory or 'data/' folder,", file=sys.stderr)
        print("or specify the path with --data-path <path/to/creditcard.csv>.", file=sys.stderr)
        sys.exit(1)

    run_eda(target_path)


if __name__ == "__main__":
    main()
