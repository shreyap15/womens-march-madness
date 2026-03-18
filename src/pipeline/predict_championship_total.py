from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.load_data import load_raw
from src.pipeline.build_dataset import build_team_features
from src.models.ensemble import blend_predictions
import joblib


@dataclass(frozen=True)
class MatchupTotals:
    expected_total: float
    expected_possessions: float


def _expected_total(team_a: pd.Series, team_b: pd.Series) -> MatchupTotals:
    """
    Estimate combined score using pace and efficiency.
    Off/Def ratings are points per possession.
    """
    poss_a = team_a.get("PossPerGame", np.nan)
    poss_b = team_b.get("PossPerGame", np.nan)
    expected_poss = np.nanmean([poss_a, poss_b])

    off_a = team_a.get("OffRtg", np.nan)
    def_a = team_a.get("DefRtg", np.nan)
    off_b = team_b.get("OffRtg", np.nan)
    def_b = team_b.get("DefRtg", np.nan)

    points_a = expected_poss * np.nanmean([off_a, def_b])
    points_b = expected_poss * np.nanmean([off_b, def_a])
    return MatchupTotals(expected_total=float(points_a + points_b), expected_possessions=float(expected_poss))


def _matchup_features(team_a: pd.Series, team_b: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    row = {}
    for col in feature_cols:
        if col.endswith("_diff"):
            base = col.replace("_diff", "")
            row[col] = team_a.get(base, np.nan) - team_b.get(base, np.nan)
        else:
            row[col] = np.nan
    return pd.DataFrame([row])


def _predict_probs(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    preds = {
        "logistic": bundle["logistic"].predict_proba(X)[:, 1],
        "rf": bundle["rf_pipeline"].predict_proba(X)[:, 1],
    }
    # Calibrate and use logistic-only to align with submission strategy
    if bundle.get("cal_log_type") == "platt":
        return bundle["cal_log"].predict_proba(preds["logistic"].reshape(-1, 1))[:, 1]
    return bundle["cal_log"].transform(preds["logistic"])


def _predict_total_from_bracket(
    season: int,
    team_features: pd.DataFrame,
    bracket_path: str,
) -> pd.DataFrame | None:
    if season != 2026:
        return None
    try:
        bracket = pd.read_csv(bracket_path)
    except FileNotFoundError:
        return None
    final_row = bracket[bracket["Slot"] == "R6CH"]
    if final_row.empty:
        return None
    final_row = final_row.iloc[0]
    # Use actual finalists from the bracket results row
    team_a = final_row.get("StrongTeamID")
    team_b = final_row.get("WeakTeamID")
    if pd.isna(team_a) or pd.isna(team_b):
        return None

    team_features = team_features.set_index("TeamID")
    ta = team_features.loc[int(team_a)]
    tb = team_features.loc[int(team_b)]
    totals = _expected_total(ta, tb)
    out = pd.DataFrame(
        [
            {
                "Season": season,
                "FinalSlot": "R6CH",
                "ExpectedCombinedScore": round(totals.expected_total, 2),
                "MostLikelyTeamA": int(team_a),
                "MostLikelyTeamB": int(team_b),
                "MostLikelyPairTotal": round(totals.expected_total, 2),
            }
        ]
    )
    return out


def _build_win_prob_lookup(
    bundle: dict, team_features: pd.DataFrame, team_ids: list[int]
) -> Dict[Tuple[int, int], float]:
    feature_cols = bundle["feature_cols"]
    t = team_features.set_index("TeamID")
    rows = []
    pairs = []
    for i, a in enumerate(team_ids):
        for b in team_ids:
            if a == b:
                continue
            ta = t.loc[a]
            tb = t.loc[b]
            rows.append(_matchup_features(ta, tb, feature_cols).iloc[0].to_dict())
            pairs.append((a, b))
    X = pd.DataFrame(rows)
    probs = _predict_probs(bundle, X)
    return {pair: float(p) for pair, p in zip(pairs, probs)}


def _slot_round(slot: str) -> int:
    if isinstance(slot, str) and slot.startswith("R") and len(slot) >= 2:
        return int(slot[1])
    return 0


def _build_slot_distributions(
    slots: pd.DataFrame,
    seed_to_team: Dict[str, int],
    win_prob: Dict[Tuple[int, int], float],
    team_features: pd.DataFrame,
) -> Tuple[Dict[str, Dict[int, float]], str]:
    slots = slots.copy()
    slots["Round"] = slots["Slot"].apply(_slot_round)
    slots = slots.sort_values("Round")

    dist: Dict[str, Dict[int, float]] = {}

    def resolve(seed_or_slot: str) -> Dict[int, float]:
        if seed_or_slot in dist:
            return dist[seed_or_slot]
        if seed_or_slot in seed_to_team:
            return {seed_to_team[seed_or_slot]: 1.0}
        return {}

    for _, row in slots.iterrows():
        slot = row["Slot"]
        strong = row["StrongSeed"]
        weak = row["WeakSeed"]
        strong_dist = resolve(strong)
        weak_dist = resolve(weak)
        out: Dict[int, float] = {}
        for team_a, p_a in strong_dist.items():
            for team_b, p_b in weak_dist.items():
                p_win = win_prob[(team_a, team_b)]
                out[team_a] = out.get(team_a, 0.0) + p_a * p_b * p_win
                out[team_b] = out.get(team_b, 0.0) + p_a * p_b * (1.0 - p_win)
        dist[slot] = out

    final_slot = slots.sort_values("Round").iloc[-1]["Slot"]
    return dist, final_slot


def predict_championship_total(
    season: int, out_path: str = "submissions/women/2026/championship_total.csv"
) -> None:
    raw = load_raw()
    slots = raw["tourney_slots"]
    seeds = raw["seeds"]

    if season not in slots["Season"].unique():
        if season != 2026:
            season = int(slots["Season"].max())
    if season not in seeds["Season"].unique():
        if season != 2026:
            season = int(seeds["Season"].max())

    slots = slots[slots["Season"] == season].copy()
    seeds = seeds[seeds["Season"] == season].copy()

    seed_to_team = dict(zip(seeds["Seed"], seeds["TeamID"]))

    team_features = build_team_features(raw)
    team_features = team_features[team_features["Season"] == season].copy()
    team_features["Elo"] = team_features["Elo"].fillna(1500.0)

    bundle = joblib.load("models/saved_models.pkl")

    # If we have a bracket result for 2026, align the total to that final matchup
    aligned = _predict_total_from_bracket(
        season,
        team_features,
        bracket_path="submissions/women/2026/bracket_2026_results.csv",
    )
    if aligned is not None:
        aligned.to_csv(out_path, index=False)
        return

    team_ids = sorted(seeds["TeamID"].unique().tolist())
    win_prob = _build_win_prob_lookup(bundle, team_features, team_ids)

    dist, final_slot = _build_slot_distributions(slots, seed_to_team, win_prob, team_features)
    final_row = slots[slots["Slot"] == final_slot].iloc[0]
    strong_dist = dist[final_row["StrongSeed"]]
    weak_dist = dist[final_row["WeakSeed"]]

    team_features = team_features.set_index("TeamID")
    expected_total = 0.0
    best_pair = None
    best_pair_prob = 0.0

    for team_a, p_a in strong_dist.items():
        for team_b, p_b in weak_dist.items():
            ta = team_features.loc[team_a]
            tb = team_features.loc[team_b]
            totals = _expected_total(ta, tb)
            pair_prob = p_a * p_b
            expected_total += pair_prob * totals.expected_total
            if pair_prob > best_pair_prob:
                best_pair_prob = pair_prob
                best_pair = (team_a, team_b, totals.expected_total)

    out = pd.DataFrame(
        [
            {
                "Season": season,
                "FinalSlot": final_slot,
                "ExpectedCombinedScore": round(expected_total, 2),
                "MostLikelyTeamA": best_pair[0] if best_pair else None,
                "MostLikelyTeamB": best_pair[1] if best_pair else None,
                "MostLikelyPairTotal": round(best_pair[2], 2) if best_pair else None,
            }
        ]
    )
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    predict_championship_total(season=2026, out_path="submissions/women/2026/championship_total.csv")
