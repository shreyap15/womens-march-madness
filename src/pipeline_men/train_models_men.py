from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from src.models.logistic_model import LogisticConfig, train_logistic
from src.models.ensemble import blend_predictions


TRAIN_SEASONS = list(range(2003, 2020))
VAL_SEASONS = [2020, 2021]
HOLDOUT_SEASONS = [2022, 2023]


def _split_by_season(df: pd.DataFrame):
    train = df[df["Season"].isin(TRAIN_SEASONS)]
    val = df[df["Season"].isin(VAL_SEASONS)]
    holdout = df[df["Season"].isin(HOLDOUT_SEASONS)]
    return train, val, holdout


def train_models(dataset_path: str = "data/processed/men/training_dataset_men.csv") -> None:
    df = pd.read_csv(dataset_path)
    preselect_base = [
        "Elo",
        "AdjNetRtg",
        "NetRtg",
        "NetRtgStd",
        "NetRtgConf",
        "GamesPlayed",
        "eFG",
        "TS",
        "OREB_rate",
        "DREB_rate",
        "TO_rate",
        "AST_TO",
        "WinPct",
        "MarginAvg",
        "MarginStd",
        "CloseWinPct",
        "HomeWinPct",
        "AwayWinPct",
        "NeutralWinPct",
        "KenPomNetRtg",
        "KenPomORtg",
        "KenPomDRtg",
        "KenPomAdjT",
        "KenPomLuck",
        "KenPomOdds",
        "SeedNum",
        "SeedPower",
        "OppNetRtg",
        "OppElo",
        "SeedMatchupUpsetRate",
    ]
    feature_cols = [f"{c}_diff" for c in preselect_base if f"{c}_diff" in df.columns]
    for extra in ["H2HGames", "H2HWinPct", "H2HMargin"]:
        if extra in df.columns:
            feature_cols.append(extra)

    train, val, _ = _split_by_season(df)
    X_train = train[feature_cols]
    y_train = train["Target"]
    X_val = val[feature_cols]
    y_val = val["Target"]

    # Logistic with light regularization
    log_model, _ = train_logistic(
        train,
        feature_cols,
        "Target",
        config=LogisticConfig(max_iter=500, C=0.2),
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        random_state=42,
        n_jobs=1,
    )
    rf_pipe = SimpleImputer(strategy="median")
    rf_pipeline = Pipeline([("imputer", rf_pipe), ("rf", rf)])
    rf_pipeline.fit(X_train, y_train)

    log_val = log_model.predict_proba(X_val)[:, 1]
    rf_val = rf_pipeline.predict_proba(X_val)[:, 1]
    cal_log = IsotonicRegression(out_of_bounds="clip")
    cal_log.fit(log_val, y_val)
    cal_rf = IsotonicRegression(out_of_bounds="clip")
    cal_rf.fit(rf_val, y_val)

    preds = {
        "logistic": cal_log.transform(log_val),
        "rf": cal_rf.transform(rf_val),
    }

    def _optimize_weights(preds_dict: dict, y_true: pd.Series) -> dict:
        keys = list(preds_dict.keys())
        P = np.vstack([preds_dict[k] for k in keys]).T

        def loss(w: np.ndarray) -> float:
            w = np.clip(w, 0.0, 1.0)
            w = w / max(w.sum(), 1e-12)
            p = np.clip(P @ w, 1e-6, 1 - 1e-6)
            return log_loss(y_true, p)

        init = np.ones(len(keys)) / len(keys)
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * len(keys)
        result = minimize(loss, init, bounds=bounds, constraints=cons, method="SLSQP")
        w = result.x if result.success else init
        w = np.clip(w, 0.0, 1.0)
        w = w / max(w.sum(), 1e-12)
        return {k: float(w[i]) for i, k in enumerate(keys)}

    corr = pd.DataFrame(preds).corr()
    corr.to_csv("data/processed/men/model_corr_men.csv")

    weights = _optimize_weights(preds, y_val)

    meta_features = list(preds.keys())
    Z_val = pd.DataFrame(preds)[meta_features].values
    meta = LogisticRegression(C=10.0, max_iter=500)
    oof = np.zeros(len(y_val))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(Z_val):
        meta_fold = LogisticRegression(C=10.0, max_iter=500)
        meta_fold.fit(Z_val[train_idx], y_val.iloc[train_idx])
        oof[val_idx] = meta_fold.predict_proba(Z_val[val_idx])[:, 1]
    meta.fit(Z_val, y_val)

    bundle = {
        "feature_cols": feature_cols,
        "logistic": log_model,
        "rf": rf,
        "rf_imputer": rf_pipe,
        "rf_pipeline": rf_pipeline,
        "cal_log": cal_log,
        "cal_rf": cal_rf,
        "weights": weights,
        "meta_features": meta_features,
        "meta_model": meta,
    }

    joblib.dump(bundle, "models/saved_models_men.pkl")

    _, _, holdout = _split_by_season(df)

    def _eval(split_name: str, split_df: pd.DataFrame) -> list[dict]:
        X = split_df[feature_cols]
        y = split_df["Target"]
        log_p = log_model.predict_proba(X)[:, 1]
        rf_p = rf_pipeline.predict_proba(X)[:, 1]
        split_preds = {
            "logistic": cal_log.transform(log_p),
            "rf": cal_rf.transform(rf_p),
        }
        blended_local = np.asarray(blend_predictions(split_preds, weights=weights))
        Z = pd.DataFrame(split_preds)[meta_features].values
        ensemble = meta.predict_proba(Z)[:, 1]
        if split_name == "validation":
            ensemble = oof
        split_preds["ensemble"] = ensemble

        rows = []
        for name, p in split_preds.items():
            p = np.clip(p, 1e-6, 1 - 1e-6)
            rows.append(
                {
                    "split": split_name,
                    "model": name,
                    "accuracy": accuracy_score(y, p >= 0.5),
                    "log_loss": log_loss(y, p),
                }
            )
        return rows

    metrics_rows = _eval("validation", val) + _eval("holdout", holdout)
    metrics = pd.DataFrame(metrics_rows).sort_values(["split", "model"])
    metrics.to_csv("data/processed/men/model_metrics_men.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    train_models()
