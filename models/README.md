# Models Directory

This directory stores trained model artifacts, weights, and evaluation metadata for fraud detection models.

### Expected Artifacts:
- `xgboost_baseline.json` / `xgboost_baseline.joblib`: Supervised baseline XGBoost classifier with class weighting (`scale_pos_weight`).
- `model_metadata.json`: Feature schema, training window config, scaling factors, and evaluation metrics (PR-AUC, ROC-AUC, Recall).
