from __future__ import annotations

import itertools
import joblib
import numpy as np
import pandas as pd

from src.pipeline_men.load_data_men import load_raw
from src.pipeline_men.build_dataset_men import build_team_features


def _team_pairs(team_ids):
    for a, b in itertools.combinations(team_ids, 2):
        yield a, b


def _build_matchups(team_features: pd.DataFrame, season: int, base_cols: list[str]) -> pd.DataFrame:
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
        for col in base_cols:
            row[f"{col}_diff"] = ta[col] - tb[col]
        rows.append(row)
    return pd.DataFrame(rows)


def _adjust_probs(matchup: pd.DataFrame, preds: np.ndarray, eps: float = 0.0005) -> np.ndarray:
    adjusted = np.clip(preds, eps, 1 - eps)
    tie_mask = np.isclose(adjusted, 0.5)
    if tie_mask.any():
        a_ids = matchup["TeamA"].astype(int).values
        b_ids = matchup["TeamB"].astype(int).values
        nudge = np.where((a_ids + b_ids) % 2 == 0, eps, -eps)
        adjusted = adjusted + tie_mask * nudge
        adjusted = np.clip(adjusted, eps, 1 - eps)
    return adjusted


def generate_predictions(
    season: int,
    out_path: str = "submissions/men/2026/submission.csv",
) -> None:
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
    if "Elo" in team_features.columns:
        team_features["Elo"] = team_features["Elo"].fillna(1500.0)

    bundle = joblib.load("models/saved_models_men.pkl")
    feature_cols = bundle["feature_cols"]
    diff_cols = [c for c in feature_cols if c.endswith("_diff")]
    base_cols = [c.replace("_diff", "") for c in diff_cols]

    matchup = _build_matchups(team_features, season, base_cols)
    for col in feature_cols:
        if col not in matchup.columns:
            matchup[col] = np.nan
    X = matchup[feature_cols]

    log_p = bundle["logistic"].predict_proba(X)[:, 1]
    calibrated = bundle["cal_log"].transform(log_p)
    calibrated = _adjust_probs(matchup, calibrated)

    ids = matchup.apply(lambda r: f"{season}_{int(r['TeamA'])}_{int(r['TeamB'])}", axis=1)
    submission = pd.DataFrame({"ID": ids, "Pred": calibrated})
    submission.to_csv(out_path, index=False)

    preds_out = pd.DataFrame({"ID": ids, "Pred": calibrated})
    preds_out.to_csv("submissions/men/2026/MNCAATourneyPredictions_with_preds.csv", index=False)


if __name__ == "__main__":
    generate_predictions(season=2026)
