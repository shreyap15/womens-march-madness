from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from src.models.logistic_model import LogisticConfig, train_logistic
from src.models.xgboost_model import train_xgboost
from src.models.ensemble import blend_predictions


TRAIN_SEASONS = list(range(2010, 2022))
VAL_SEASONS = [2022, 2023]
HOLDOUT_SEASONS = [2024, 2025]


def _split_by_season(df: pd.DataFrame):
    train = df[df["Season"].isin(TRAIN_SEASONS)]
    val = df[df["Season"].isin(VAL_SEASONS)]
    holdout = df[df["Season"].isin(HOLDOUT_SEASONS)]
    return train, val, holdout


def train_models(dataset_path: str = "data/processed/training_dataset.csv") -> None:
    df = pd.read_csv(dataset_path)
    feature_cols = [c for c in df.columns if c.endswith("_diff")]

    train, val, _ = _split_by_season(df)
    X_train = train[feature_cols]
    y_train = train["Target"]
    X_val = val[feature_cols]
    y_val = val["Target"]

    # Logistic regression baseline
    log_model, _ = train_logistic(
        train,
        feature_cols,
        "Target",
        config=LogisticConfig(max_iter=500),
    )

    # Random forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=1,
    )
    rf_pipe = SimpleImputer(strategy="median")
    rf.fit(rf_pipe.fit_transform(X_train), y_train)

    # XGBoost
    xgb_model = None
    try:
        xgb_model, _ = train_xgboost(train, feature_cols, "Target")
    except ImportError:
        xgb_model = None

    # Validation predictions
    preds = {
        "logistic": log_model.predict_proba(X_val)[:, 1],
        "rf": rf.predict_proba(rf_pipe.transform(X_val))[:, 1],
        "elo": 0.5 + 0.5 * np.tanh(X_val["Elo_diff"] / 400.0),
    }
    weights = {"logistic": 0.45, "rf": 0.30, "elo": 0.25}
    if xgb_model is not None:
        preds["xgb"] = xgb_model.predict_proba(X_val)[:, 1]
        weights = {"xgb": 0.50, "logistic": 0.25, "rf": 0.15, "elo": 0.10}

    blended = np.asarray(blend_predictions(preds, weights=weights))

    # Platt scaling on blended outputs
    calibrator = LogisticRegression()
    calibrator.fit(blended.reshape(-1, 1), y_val)

    bundle = {
        "feature_cols": feature_cols,
        "logistic": log_model,
        "rf": rf,
        "rf_imputer": rf_pipe,
        "xgb": xgb_model,
        "calibrator": calibrator,
        "weights": weights,
    }

    joblib.dump(bundle, "models/saved_models.pkl")


if __name__ == "__main__":
    train_models()
