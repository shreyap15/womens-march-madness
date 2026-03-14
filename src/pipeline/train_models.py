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
    # Pre-selection feature set (only features available before seeds/ESPN/FTE)
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
        "Last5NetRtg",
        "Last10NetRtg",
        "TrendNetRtg",
        "RoadPerformanceGap",
        "QualityWins",
        "EloTrajectory",
        "EloPeak",
        "EloCurrentVsPeak",
        "WinPctStd3",
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

    # Logistic regression baseline (tune C on validation with calibration)
    c_grid = [0.05, 0.1, 0.2, 0.5, 1.0]
    best_log = None
    best_log_val = None
    best_log_cal = None
    best_log_cal_type = None
    best_log_loss = float("inf")
    best_log_c = None
    for c in c_grid:
        log_model_tmp, _ = train_logistic(
            train,
            feature_cols,
            "Target",
            config=LogisticConfig(max_iter=500, C=c),
        )
        log_val_tmp = log_model_tmp.predict_proba(X_val)[:, 1]
        log_iso = IsotonicRegression(out_of_bounds="clip")
        log_iso.fit(log_val_tmp, y_val)
        log_sig = LogisticRegression(max_iter=1000)
        log_sig.fit(log_val_tmp.reshape(-1, 1), y_val)

        p_iso = np.clip(log_iso.transform(log_val_tmp), 1e-6, 1 - 1e-6)
        p_sig = np.clip(
            log_sig.predict_proba(log_val_tmp.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6
        )
        ll_iso = log_loss(y_val, p_iso)
        ll_sig = log_loss(y_val, p_sig)
        if ll_sig < ll_iso:
            ll = ll_sig
            cal = log_sig
            cal_type = "platt"
            val_pred = p_sig
        else:
            ll = ll_iso
            cal = log_iso
            cal_type = "isotonic"
            val_pred = p_iso

        if ll < best_log_loss:
            best_log_loss = ll
            best_log = log_model_tmp
            best_log_cal = cal
            best_log_cal_type = cal_type
            best_log_val = val_pred
            best_log_c = c

    log_model = best_log
    log_cal_type = best_log_cal_type
    cal_log = best_log_cal
    log_cal_pred = best_log_val

    # Random forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=1,
    )
    rf_pipe = SimpleImputer(strategy="median")
    rf_pipeline = Pipeline([("imputer", rf_pipe), ("rf", rf)])
    rf_pipeline.fit(X_train, y_train)


    # Calibrate each model on validation set (compare isotonic vs sigmoid)
    rf_val = rf_pipeline.predict_proba(X_val)[:, 1]

    cal_rf_iso = IsotonicRegression(out_of_bounds="clip")
    cal_rf_iso.fit(rf_val, y_val)
    cal_rf_sig = LogisticRegression(max_iter=1000)
    cal_rf_sig.fit(rf_val.reshape(-1, 1), y_val)

    def _pick_calibrator(
        base_probs: np.ndarray,
        iso: IsotonicRegression,
        sig: LogisticRegression,
    ) -> tuple[str, object, np.ndarray]:
        p_iso = np.clip(iso.transform(base_probs), 1e-6, 1 - 1e-6)
        p_sig = np.clip(
            sig.predict_proba(base_probs.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6
        )
        ll_iso = log_loss(y_val, p_iso)
        ll_sig = log_loss(y_val, p_sig)
        if ll_sig < ll_iso:
            return "platt", sig, p_sig
        return "isotonic", iso, p_iso

    rf_cal_type, cal_rf, rf_cal_pred = _pick_calibrator(rf_val, cal_rf_iso, cal_rf_sig)

    # Validation predictions (calibrated)
    preds = {
        "logistic": log_cal_pred,
        "rf": rf_cal_pred,
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

    # Correlation matrix on validation predictions
    corr = pd.DataFrame(preds).corr()
    corr.to_csv("data/processed/model_corr.csv")

    weights = _optimize_weights(preds, y_val)

    # Stacking meta-learner (trained on validation preds with CV to avoid leakage)
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
        "cal_log_type": log_cal_type,
        "cal_rf_type": rf_cal_type,
        "logistic_C": best_log_c,
        "weights": weights,
        "meta_features": meta_features,
        "meta_model": meta,
    }

    joblib.dump(bundle, "models/saved_models.pkl")

    # Metrics on validation and holdout splits
    _, _, holdout = _split_by_season(df)

    def _eval(split_name: str, split_df: pd.DataFrame) -> list[dict]:
        X = split_df[feature_cols]
        y = split_df["Target"]
        log_p = log_model.predict_proba(X)[:, 1]
        rf_p = rf_pipeline.predict_proba(X)[:, 1]
        if log_cal_type == "platt":
            log_cal = cal_log.predict_proba(log_p.reshape(-1, 1))[:, 1]
        else:
            log_cal = cal_log.transform(log_p)
        if rf_cal_type == "platt":
            rf_cal = cal_rf.predict_proba(rf_p.reshape(-1, 1))[:, 1]
        else:
            rf_cal = cal_rf.transform(rf_p)
        split_preds = {
            "logistic": log_cal,
            "rf": rf_cal,
        }
        # Simple blend without meta-learner
        blend_60_40 = 0.60 * log_cal + 0.40 * rf_cal
        split_preds["blend60_40"] = blend_60_40

        # Clipped variants for log loss protection
        split_preds["logistic_clip"] = np.clip(log_cal, 0.05, 0.95)
        split_preds["blend60_40_clip"] = np.clip(blend_60_40, 0.05, 0.95)
        blended_local = np.asarray(blend_predictions(split_preds, weights=weights))
        Z = pd.DataFrame(split_preds)[meta_features].values
        ensemble = meta.predict_proba(Z)[:, 1]
        # Use out-of-fold ensemble for validation to avoid leakage
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
    metrics.to_csv("data/processed/model_metrics.csv", index=False)
    metrics.to_csv("data/processed/model_metrics_all_models.csv", index=False)
    print(f"Selected logistic C={best_log_c}, calibration={log_cal_type}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    train_models()
