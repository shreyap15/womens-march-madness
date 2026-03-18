from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.load_data import load_raw
from src.pipeline.build_dataset import build_team_features


def _seed_num(seed: str | float | int | None) -> float:
    if seed is None or (isinstance(seed, float) and np.isnan(seed)):
        return np.nan
    if isinstance(seed, (int, float)):
        return float(seed)
    # seed codes like W01, X16, R1W1, X11a
    m = re.search(r"(\d{1,2})", str(seed))
    return float(m.group(1)) if m else np.nan


def _winner_from_prob(
    team_a: Tuple[int, str],
    team_b: Tuple[int, str],
    prob_a: float | None,
    seed_a: float,
    seed_b: float,
) -> Tuple[int, str, float]:
    if prob_a is None or np.isnan(prob_a):
        if not np.isnan(seed_a) and not np.isnan(seed_b):
            return (team_a[0], team_a[1], 0.5) if seed_a < seed_b else (team_b[0], team_b[1], 0.5)
        return team_a[0], team_a[1], 0.5
    return (team_a[0], team_a[1], prob_a) if prob_a >= 0.5 else (team_b[0], team_b[1], 1 - prob_a)


def predict_full_bracket(
    bracket_path: str = "submissions/women/2026/bracket_2026.csv",
    preds_path: str = "submissions/women/2026/WNCAATourneyPredictions_with_preds.csv",
    out_path: str = "submissions/women/2026/bracket_2026_results.csv",
    viz_path: str = "submissions/women/2026/bracket_2026_visual.md",
) -> None:
    bracket = pd.read_csv(bracket_path)
    preds = pd.read_csv(preds_path)
    raw = load_raw()
    team_features = build_team_features(raw)
    team_features = team_features[team_features["Season"] == 2026].set_index("TeamID")

    pred_map = dict(zip(preds["ID"], preds["Pred"]))

    def matchup_prob(a_id: int, b_id: int) -> float | None:
        lo, hi = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        key = f"2026_{lo}_{hi}"
        p = pred_map.get(key)
        if p is None:
            return None
        return p if a_id == lo else 1 - p

    bracket["_round_sort"] = bracket["Round"].fillna(-1)
    bracket = (
        bracket.sort_values(["_round_sort", "Slot"])
        .drop(columns=["_round_sort"])
        .reset_index(drop=True)
    )

    slot_winner: Dict[str, Dict[str, object]] = {}
    out_rows = []

    for _, row in bracket.iterrows():
        strong_seed = row.get("StrongSeed")
        weak_seed = row.get("WeakSeed")

        strong_id = row.get("StrongTeamID")
        strong_name = row.get("StrongTeamName")
        if pd.isna(strong_id) and isinstance(strong_seed, str) and strong_seed in slot_winner:
            strong_id = slot_winner[strong_seed]["TeamID"]
            strong_name = slot_winner[strong_seed]["TeamName"]

        weak_id = row.get("WeakTeamID")
        weak_name = row.get("WeakTeamName")
        if pd.isna(weak_id) and isinstance(weak_seed, str) and weak_seed in slot_winner:
            weak_id = slot_winner[weak_seed]["TeamID"]
            weak_name = slot_winner[weak_seed]["TeamName"]

        if pd.isna(strong_id) or pd.isna(weak_id):
            out_rows.append(
                {
                    **row,
                    "WinnerTeamID": np.nan,
                    "WinnerTeamName": np.nan,
                    "WinnerSeed": np.nan,
                    "WinnerProb": np.nan,
                }
            )
            continue

        strong_id = int(strong_id)
        weak_id = int(weak_id)
        team_a = (strong_id, strong_name)
        team_b = (weak_id, weak_name)
        prob_a = matchup_prob(strong_id, weak_id)

        # Deterministic tiebreak for near-coinflip cases
        if prob_a is not None and 0.49 <= prob_a <= 0.51:
            elo_a = team_features.at[strong_id, "Elo"] if strong_id in team_features.index else np.nan
            elo_b = team_features.at[weak_id, "Elo"] if weak_id in team_features.index else np.nan
            adj_a = team_features.at[strong_id, "AdjNetRtg"] if strong_id in team_features.index else np.nan
            adj_b = team_features.at[weak_id, "AdjNetRtg"] if weak_id in team_features.index else np.nan
            if not np.isnan(elo_a) and not np.isnan(elo_b):
                elo_diff = elo_a - elo_b
            else:
                elo_diff = 0.0
            if not np.isnan(adj_a) and not np.isnan(adj_b):
                adj_diff = adj_a - adj_b
            else:
                adj_diff = 0.0
            prob_a = float(np.clip(prob_a + (elo_diff / 400.0) * 0.15 + adj_diff * 0.05, 0.0, 1.0))

        seed_a = _seed_num(strong_seed)
        seed_b = _seed_num(weak_seed)
        winner_id, winner_name, winner_prob = _winner_from_prob(team_a, team_b, prob_a, seed_a, seed_b)
        winner_seed = seed_a if winner_id == strong_id else seed_b

        slot_winner[row["Slot"]] = {
            "TeamID": winner_id,
            "TeamName": winner_name,
            "Seed": winner_seed,
            "Prob": winner_prob,
        }

        out_rows.append(
            {
                **row,
                "StrongTeamID": strong_id,
                "StrongTeamName": strong_name,
                "WeakTeamID": weak_id,
                "WeakTeamName": weak_name,
                "WinnerTeamID": winner_id,
                "WinnerTeamName": winner_name,
                "WinnerSeed": winner_seed,
                "WinnerProb": winner_prob,
            }
        )

    out = pd.DataFrame(out_rows)
    out.to_csv(out_path, index=False)

    viz_lines = ["# 2026 NCAA W Bracket (Model Picks)", ""]
    for rnd, label in [
        (1.0, "Round 1"),
        (2.0, "Round 2"),
        (3.0, "Sweet 16"),
        (4.0, "Elite 8"),
        (5.0, "Final Four"),
        (6.0, "Championship"),
    ]:
        viz_lines.append(f"## {label}")
        subset = out[out["Round"] == rnd]
        if subset.empty:
            viz_lines.append("_No games_")
            viz_lines.append("")
            continue
        for _, r in subset.iterrows():
            a = r.get("StrongTeamName")
            b = r.get("WeakTeamName")
            w = r.get("WinnerTeamName")
            p = r.get("WinnerProb")
            if pd.isna(a) or pd.isna(b) or pd.isna(w):
                continue
            viz_lines.append(f"- {r['Slot']}: {a} vs {b} -> **{w}** (p={p:.4f})")
        viz_lines.append("")

    with open(viz_path, "w", encoding="ascii") as f:
        f.write("\n".join(viz_lines))


if __name__ == "__main__":
    predict_full_bracket()
