# Models Directory

This directory stores trained model artifacts, weights, and evaluation metadata for fraud detection models.

### Expected Artifacts:
- `xgboost_baseline.json` / `xgboost_baseline.joblib`: Supervised baseline XGBoost classifier with class weighting (`scale_pos_weight`).
- `isolation_forest.joblib`: Unsupervised baseline Isolation Forest model trained strictly on normal transactions.
- `model_metadata.json` / `isolation_forest_metadata.json`: Feature schemas, training window configs, and evaluation metrics (PR-AUC, ROC-AUC, Recall).
