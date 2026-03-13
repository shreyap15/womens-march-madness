from __future__ import annotations

import itertools
import joblib
import numpy as np
import pandas as pd

from src.data.load_data import load_raw
from src.pipeline.build_dataset import build_team_features
from src.models.ensemble import blend_predictions


def _team_pairs(team_ids):
    for a, b in itertools.combinations(team_ids, 2):
        yield a, b


def _build_matchups(team_features: pd.DataFrame, season: int, feature_cols: list[str]) -> pd.DataFrame:
    t = team_features[team_features["Season"] == season].set_index("TeamID")
    rows = []
    for a, b in _team_pairs(t.index.tolist()):
        ta = t.loc[a]
        tb = t.loc[b]
        row = {
            "Season": season,
            "TeamA": a,
            "TeamB": b,
        }
        for col in feature_cols:
            row[f"{col}_diff"] = ta[col] - tb[col]
        rows.append(row)
    return pd.DataFrame(rows)


def generate_predictions(season: int, out_path: str = "submissions/submission.csv") -> None:
    raw = load_raw()
    team_features = build_team_features(raw)
    all_teams = raw["teams"][["TeamID"]].drop_duplicates()
    base = all_teams.assign(Season=season)
    team_features = (
        base.merge(
            team_features[team_features["Season"] == season],
            on=["Season", "TeamID"],
            how="left",
        )
        .copy()
    )
    # Ensure Elo is defined for the Elo-only signal
    if "Elo" in team_features.columns:
        team_features["Elo"] = team_features["Elo"].fillna(1500.0)

    bundle = joblib.load("models/saved_models.pkl")
    feature_cols = bundle["feature_cols"]
    matchup = _build_matchups(team_features, season, [c.replace("_diff", "") for c in feature_cols])
    X = matchup[feature_cols]

    preds = {
        "logistic": bundle["logistic"].predict_proba(X)[:, 1],
        "rf": bundle["rf"].predict_proba(bundle["rf_imputer"].transform(X))[:, 1],
        "elo": 0.5 + 0.5 * np.tanh(X["Elo_diff"] / 400.0),
    }
    if bundle.get("xgb") is not None:
        preds["xgb"] = bundle["xgb"].predict_proba(X)[:, 1]

    blended = np.asarray(blend_predictions(preds, weights=bundle.get("weights")))
    calibrated = bundle["calibrator"].predict_proba(blended.reshape(-1, 1))[:, 1]

    ids = matchup.apply(lambda r: f"{season}_{int(r['TeamA'])}_{int(r['TeamB'])}", axis=1)
    submission = pd.DataFrame({"ID": ids, "Pred": calibrated})
    submission.to_csv(out_path, index=False)


if __name__ == "__main__":
    generate_predictions(season=2026)
