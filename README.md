Women’s March Madness Prediction Pipeline

Quick start:
1. Put Kaggle CSVs into data/raw using the standard file names.
2. Build dataset: python -m src.pipeline.build_dataset
3. Train models: python -m src.pipeline.train_models
4. Generate predictions: python -m src.pipeline.generate_predictions

Notes:
- Pre-2010 rows keep advanced features as NaN.
- XGBoost is optional but recommended for best accuracy.
