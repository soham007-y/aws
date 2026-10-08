#!/usr/bin/env python3
"""
02_preprocess.py - Preprocessing, Feature Scaling, and SMOTE + Class Weighting
Kaggle Credit Card Fraud Detection Dataset
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


# ==============================================================================
# DETAILED INLINE RATIONALE: WHY NAIVE OVERSAMPLING FAILS FOR EXTREME FRAUD IMBALANCE
# ==============================================================================
#
# In extreme financial fraud datasets (e.g., Credit Card Fraud where fraud is ~0.17%
# or ~1:578 ratio), naive Random Over-Sampling (ROS) simply duplicates existing
# fraudulent transactions with replacement until parity (50:50) is reached.
#
# This naive approach fundamentally fails in production fraud systems for several reasons:
#
# 1. Severe Overfitting & Exact Memorization:
#    - Duplicating ~350-400 fraud instances hundreds of times forces tree-based
#      ensembles (XGBoost, LightGBM, Random Forest) and deep nets to build ultra-specific
#      decision leaves around exact duplicate coordinates in feature space.
#    - The model memorizes exact noise and idiosyncrasies of historical frauds rather than
#      learning generalizable decision boundaries across feature distributions.
#
# 2. Catastrophic False Positive Rates (FPR) in Production:
#    - Artificially forcing a 50:50 balance dramatically distorts the empirical prior
#      probability P(Fraud) from 0.0017 to 0.50.
#    - Calibrated posterior probabilities P(Fraud | X) become massively inflated.
#    - When deployed against millions of legitimate daily transactions, the model
#      triggers thousands of false alarms, overwhelming fraud investigation teams
#      and alienating genuine cardholders.
#
# 3. Inability to Bridge Sparse Minority Space:
#    - Naive duplication creates dense point-mass spikes at existing points without
#      exploring the manifold between fraud samples.
#    - SMOTE partially addresses this by creating synthetic samples along line segments
#      connecting k-nearest minority neighbors in feature space.
#
# 4. The Danger of Aggressive 50:50 SMOTE & Why Hybrid (SMOTE + Class Weighting) Wins:
#    - If SMOTE is used naively to synthesize fraud all the way up to 50% (generating
#      ~200,000 synthetic fraud instances), it generates points directly inside majority-class
#      dense clusters whenever minority outliers or label errors exist ("synthetic noise").
#    - Best Practice Hybrid Strategy:
#      * Use mild/moderate SMOTE (e.g., sampling_strategy=0.1, raising fraud to ~1:10) to
#        densify the sparse minority region and establish local connectivity.
#      * Combine with algorithmic minority class weighting (e.g., scale_pos_weight in XGBoost
#        or cost-sensitive loss penalties) to penalize missed frauds without polluting
#        the feature space with hundreds of thousands of artificial data points.
# ==============================================================================


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
    and a 'later_window' (last 30%) to simulate time drift without shuffling.
    """
    if "Time" in df.columns:
        df_sorted = df.sort_values(by="Time").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    split_idx = int(len(df_sorted) * train_ratio)
    training_window = df_sorted.iloc[:split_idx].copy()
    later_window = df_sorted.iloc[split_idx:].copy()

    return training_window, later_window


def preprocess_data(
    data_path: Path,
    smote_ratio: float = 0.10,
    random_state: int = 42,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """
    Loads dataset, performs chronological split, scales features, and applies
    SMOTE combined with minority class weighting on the training window only.
    """
    print("=" * 80)
    print(f"Loading dataset from: {data_path}")
    print("=" * 80)

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if "Class" not in df.columns:
        print("Error: 'Class' target column missing.", file=sys.stderr)
        sys.exit(1)

    # 1. Chronological Split (No Shuffling to prevent temporal data leakage)
    training_window, later_window = chronological_split(df, train_ratio=0.70)
    print(f"\n[Split] Chronological Split:")
    print(f"  Training Window (70%): {len(training_window):,} rows")
    print(f"  Later Window    (30%): {len(later_window):,} rows")

    # 2. Separate Features and Target
    X_train = training_window.drop(columns=["Class"]).copy()
    y_train = training_window["Class"].astype(int).values

    X_later = later_window.drop(columns=["Class"]).copy()
    y_later = later_window["Class"].astype(int).values

    print("\n[Original Training Distribution]")
    train_counts = pd.Series(y_train).value_counts().to_dict()
    print(f"  Class 0 (Legitimate): {train_counts.get(0, 0):,}")
    print(f"  Class 1 (Fraud):      {train_counts.get(1, 0):,}")
    orig_ratio = train_counts.get(0, 0) / max(train_counts.get(1, 0), 1)
    print(f"  Imbalance Ratio:      {orig_ratio:.2f} : 1")

    # 3. Feature Scaling
    # V1-V28 are PCA transformed features.
    # 'Amount' exhibits heavy right skew with extreme outliers -> RobustScaler
    # 'Time' represents elapsed seconds -> StandardScaler
    feature_names = list(X_train.columns)
    scalers: dict[str, object] = {}

    if "Amount" in X_train.columns:
        amount_scaler = RobustScaler()
        X_train["Amount"] = amount_scaler.fit_transform(X_train[["Amount"]])
        X_later["Amount"] = amount_scaler.transform(X_later[["Amount"]])
        scalers["Amount"] = amount_scaler

    if "Time" in X_train.columns:
        time_scaler = StandardScaler()
        X_train["Time"] = time_scaler.fit_transform(X_train[["Time"]])
        X_later["Time"] = time_scaler.transform(X_later[["Time"]])
        scalers["Time"] = time_scaler

    X_train_scaled = X_train.values
    X_later_scaled = X_later.values

    # 4. SMOTE on Training Window Only (Never resample test/validation data!)
    # We apply moderate SMOTE (e.g. sampling_strategy=0.10) to create synthetic connectivity
    # without overwhelming feature space with synthetic noise.
    print(f"\n[SMOTE Resampling]")
    print(f"  Target sampling_strategy: {smote_ratio} (Fraud to Legitimate ratio)")
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    resampled_counts = pd.Series(y_train_resampled).value_counts().to_dict()
    print(f"  Resampled Class 0:        {resampled_counts.get(0, 0):,}")
    print(f"  Resampled Class 1:        {resampled_counts.get(1, 0):,}")
    new_ratio = resampled_counts.get(0, 0) / max(resampled_counts.get(1, 0), 1)
    print(f"  Resampled Imbalance:      {new_ratio:.2f} : 1")

    # 5. Class Weighting Calculation
    # Compute balanced weights to adjust loss function penalties during training
    classes = np.unique(y_train_resampled)
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_resampled,
    )
    class_weight_dict = {cls: weight for cls, weight in zip(classes, balanced_weights)}

    # XGBoost scale_pos_weight formula: negative_count / positive_count
    scale_pos_weight = resampled_counts.get(0, 0) / max(resampled_counts.get(1, 0), 1)

    print("\n[Minority Class Weighting Configuration]")
    print(f"  Class Weights (balanced): {class_weight_dict}")
    print(f"  XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    # 6. Save preprocessed artifacts if output directory is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "preprocessed_data.npz"
        np.savez_compressed(
            save_path,
            X_train_resampled=X_train_resampled,
            y_train_resampled=y_train_resampled,
            X_later_scaled=X_later_scaled,
            y_later=y_later,
            scale_pos_weight=np.array([scale_pos_weight]),
        )
        print(f"\n[Artifacts] Preprocessed splits saved to: {save_path}")

    print("\n" + "=" * 80)
    print("Preprocessing & SMOTE Class Weighting Pipeline Complete.")
    print("=" * 80)

    return {
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "X_later_scaled": X_later_scaled,
        "y_later": y_later,
        "class_weight_dict": class_weight_dict,
        "scale_pos_weight": scale_pos_weight,
        "feature_names": feature_names,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Credit Card Fraud dataset with chronological split, scaling, and SMOTE class weighting"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to creditcard.csv (defaults to auto-detection)",
    )
    parser.add_argument(
        "--smote-ratio",
        type=float,
        default=0.10,
        help="SMOTE sampling strategy ratio (minority to majority, default: 0.10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save preprocessed splits (default: data/processed)",
    )
    args = parser.parse_args()

    target_path = Path(args.data_path) if args.data_path else find_dataset()

    if target_path is None or not target_path.exists():
        print("Dataset not found!", file=sys.stderr)
        print("Please place 'creditcard.csv' in the project directory or 'data/' folder,", file=sys.stderr)
        print("or specify with --data-path <path/to/creditcard.csv>.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None
    preprocess_data(
        data_path=target_path,
        smote_ratio=args.smote_ratio,
        random_state=args.random_state,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
