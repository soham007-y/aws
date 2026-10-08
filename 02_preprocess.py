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
# DETAILED INLINE RATIONALE: WHY NAIVE OVERSAMPLING ALONE IS INSUFFICIENT
# FOR EXTREME FINANCIAL FRAUD IMBALANCE
# ==============================================================================
#
# In extreme financial fraud datasets (such as Kaggle Credit Card Fraud where fraud
# constitutes only ~0.17% of transactions or a ~1:578 ratio), naive Random Over-Sampling (ROS)
# simply duplicates existing minority (fraudulent) transactions with replacement until parity
# or a higher balance is reached.
#
# This naive approach fundamentally fails in production financial fraud detection systems for several critical reasons:
#
# 1. Severe Overfitting & Exact Memorization:
#    - Duplicating ~350-400 fraud instances hundreds of times forces tree-based ensembles
#      (e.g., XGBoost, LightGBM, Random Forest) and neural networks to create ultra-specific,
#      tight decision boundaries around exact duplicate coordinates in feature space.
#    - The model memorizes exact noise and unique quirks of specific past fraud instances
#      rather than learning generalizable decision boundaries across the feature distribution.
#
# 2. Catastrophic False Positive Rates (FPR) in Production:
#    - Artificially forcing a 50:50 or heavy oversample distorts the empirical prior
#      probability P(Fraud) from 0.0017 to 0.50.
#    - Calibrated posterior probabilities P(Fraud | X) become massively inflated.
#    - When deployed against millions of legitimate daily transactions, the model triggers
#      thousands of false alarms, overwhelming fraud investigation ops and alienating genuine cardholders.
#
# 3. Inability to Explore Sparse Minority Manifold:
#    - Naive duplication creates dense point-mass spikes at existing sample locations
#      without exploring the manifold or feature space between fraud instances.
#    - SMOTE (Synthetic Minority Over-sampling Technique) addresses this by creating synthetic
#      samples along line segments connecting k-nearest minority neighbors in feature space.
#
# 4. The Superiority of Hybrid Strategy (Moderate SMOTE + Class Weighting):
#    - Aggressive 50:50 SMOTE generates synthetic noise inside dense majority clusters whenever
#      minority outliers or label noise exist.
#    - Recommended Hybrid Approach:
#      * Use mild/moderate SMOTE (e.g., sampling_strategy=0.10, raising fraud to ~1:10) to
#        densify the sparse minority region and establish local connectivity.
#      * Combine with algorithmic minority class weighting (e.g., scale_pos_weight in XGBoost
#        or cost-sensitive loss penalties) to heavily penalize missed frauds during loss optimization
#        without inflating dataset size with hundreds of thousands of artificial data points.
# ==============================================================================


def find_dataset(default_filename: str = "train_window.csv") -> Path | None:
    """Search for the dataset in common project paths."""
    candidates = [
        Path("data") / default_filename,
        Path(default_filename),
        Path("data") / "creditcard.csv",
        Path("creditcard.csv"),
        Path("..") / default_filename,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split dataset into training (first 80%) and validation (remaining 20%)
    without shuffling to simulate temporal sequence.
    """
    if "Time" in df.columns:
        df_sorted = df.sort_values(by="Time").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()

    return train_df, val_df


def preprocess_data(
    data_path: Path,
    val_ratio: float = 0.20,
    smote_ratio: float = 0.10,
    random_state: int = 42,
    output_dir: Path | None = Path("data"),
) -> dict[str, object]:
    """
    Loads data, splits into train/validation windows, scales numerical features,
    applies SMOTE and class weighting on the training split, and saves preprocessed artifacts.
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
        print("Error: 'Class' target column missing from dataset.", file=sys.stderr)
        sys.exit(1)

    # 1. Train / Validation Chronological Split
    train_ratio = 1.0 - val_ratio
    train_df, val_df = chronological_split(df, train_ratio=train_ratio)
    print(f"\n[Split] Dataset Split (Train: {train_ratio:.0%}, Val: {val_ratio:.0%}):")
    print(f"  Training Split:   {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    print(f"  Validation Split: {val_df.shape[0]:,} rows, {val_df.shape[1]} columns")

    # 2. Separate Features and Target
    X_train = train_df.drop(columns=["Class"]).copy()
    y_train = train_df["Class"].astype(int).values

    X_val = val_df.drop(columns=["Class"]).copy()
    y_val = val_df["Class"].astype(int).values

    feature_names = list(X_train.columns)

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
    scalers: dict[str, object] = {}

    if "Amount" in X_train.columns:
        amount_scaler = RobustScaler()
        X_train["Amount"] = amount_scaler.fit_transform(X_train[["Amount"]])
        X_val["Amount"] = amount_scaler.transform(X_val[["Amount"]])
        scalers["Amount"] = amount_scaler

    if "Time" in X_train.columns:
        time_scaler = StandardScaler()
        X_train["Time"] = time_scaler.fit_transform(X_train[["Time"]])
        X_val["Time"] = time_scaler.transform(X_val[["Time"]])
        scalers["Time"] = time_scaler

    X_train_scaled = X_train.values
    X_val_scaled = X_val.values

    # 4. SMOTE Resampling on Training Split Only
    print(f"\n[SMOTE Resampling]")
    print(f"  Target sampling_strategy: {smote_ratio} (Fraud to Legitimate ratio)")
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    resampled_counts = pd.Series(y_train_resampled).value_counts().to_dict()
    print(f"  Resampled Class 0:        {resampled_counts.get(0, 0):,}")
    print(f"  Resampled Class 1:        {resampled_counts.get(1, 0):,}")
    new_ratio = resampled_counts.get(0, 0) / max(resampled_counts.get(1, 0), 1)
    print(f"  Resampled Imbalance:      {new_ratio:.2f} : 1")

    # 5. Minority Class Weighting Calculation
    classes = np.unique(y_train_resampled)
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_resampled,
    )
    class_weight_dict = {cls: weight for cls, weight in zip(classes, balanced_weights)}
    scale_pos_weight = resampled_counts.get(0, 0) / max(resampled_counts.get(1, 0), 1)

    print("\n[Minority Class Weighting Configuration]")
    print(f"  Class Weights (balanced): {class_weight_dict}")
    print(f"  XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    # 6. Save preprocessed artifacts in data/ folder
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV files as requested
        train_preprocessed_df = pd.DataFrame(X_train_resampled, columns=feature_names)
        train_preprocessed_df["Class"] = y_train_resampled
        train_csv_path = output_dir / "train_preprocessed.csv"
        train_preprocessed_df.to_csv(train_csv_path, index=False)

        val_preprocessed_df = pd.DataFrame(X_val_scaled, columns=feature_names)
        val_preprocessed_df["Class"] = y_val
        val_csv_path = output_dir / "val_preprocessed.csv"
        val_preprocessed_df.to_csv(val_csv_path, index=False)

        # Compressed .npz archives for fast downstream model loading
        npz_path = output_dir / "preprocessed_data.npz"
        np.savez_compressed(
            npz_path,
            X_train_resampled=X_train_resampled,
            y_train_resampled=y_train_resampled,
            X_later_scaled=X_val_scaled,
            y_later=y_val,
            scale_pos_weight=np.array([scale_pos_weight]),
        )

        processed_dir = output_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            processed_dir / "preprocessed_data.npz",
            X_train_resampled=X_train_resampled,
            y_train_resampled=y_train_resampled,
            X_later_scaled=X_val_scaled,
            y_later=y_val,
            scale_pos_weight=np.array([scale_pos_weight]),
        )

        print(f"\n[Artifacts Saved in {output_dir}/]:")
        print(f"  - Preprocessed Training CSV:   {train_csv_path} ({train_preprocessed_df.shape})")
        print(f"  - Preprocessed Validation CSV: {val_csv_path} ({val_preprocessed_df.shape})")
        print(f"  - Compressed Archive:          {npz_path}")

    print("\n" + "=" * 80)
    print("Preprocessing & SMOTE Class Weighting Pipeline Complete.")
    print("=" * 80)

    return {
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "X_val_scaled": X_val_scaled,
        "y_val": y_val,
        "class_weight_dict": class_weight_dict,
        "scale_pos_weight": scale_pos_weight,
        "feature_names": feature_names,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Credit Card Fraud dataset with scaling, SMOTE, and class weighting"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset (defaults to auto-detecting data/train_window.csv)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation split ratio (default: 0.20)",
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
        default="data",
        help="Directory to save preprocessed splits (default: data)",
    )
    args = parser.parse_args()

    target_path = Path(args.data_path) if args.data_path else find_dataset()

    if target_path is None or not target_path.exists():
        print("Dataset not found!", file=sys.stderr)
        print("Please ensure 'data/train_window.csv' exists (run 01_eda.py first),", file=sys.stderr)
        print("or specify path with --data-path <path/to/dataset.csv>.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path("data")
    preprocess_data(
        data_path=target_path,
        val_ratio=args.val_ratio,
        smote_ratio=args.smote_ratio,
        random_state=args.random_state,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
