from __future__ import annotations

import numpy as np
import pandas as pd


def _seed_num(seed: str | float | int | None) -> float:
    if seed is None or (isinstance(seed, float) and np.isnan(seed)):
        return np.nan
    if isinstance(seed, (int, float)):
        return float(seed)
    import re
    m = re.search(r"(\d{1,2})", str(seed))
    return float(m.group(1)) if m else np.nan


def predict_full_bracket(
    bracket_path: str = "submissions/men/2026/bracket_2026.csv",
    preds_path: str = "submissions/men/2026/MNCAATourneyPredictions_with_preds.csv",
    out_path: str = "submissions/men/2026/bracket_2026_results.csv",
    viz_path: str = "submissions/men/2026/bracket_2026_visual.md",
) -> None:
    bracket = pd.read_csv(bracket_path)
    preds = pd.read_csv(preds_path)

    pred_map = dict(zip(preds["ID"], preds["Pred"]))

    def matchup_prob(a_id: int, b_id: int) -> float | None:
        lo, hi = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        key = f"2026_{lo}_{hi}"
        p = pred_map.get(key)
        if p is None:
            return None
        return p if a_id == lo else 1 - p

    if "Round" in bracket.columns:
        bracket["_round_sort"] = bracket["Round"].fillna(-1)
    else:
        bracket["_round_sort"] = bracket["Slot"].str.slice(1, 2).astype(float).fillna(-1)
    bracket = bracket.sort_values(["_round_sort", "Slot"]).drop(columns=["_round_sort"])

    slot_winner = {}
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
            out_rows.append({**row, "WinnerTeamID": np.nan, "WinnerTeamName": np.nan, "WinnerSeed": np.nan, "WinnerProb": np.nan})
            continue

        strong_id = int(strong_id)
        weak_id = int(weak_id)
        prob_a = matchup_prob(strong_id, weak_id)
        seed_a = _seed_num(strong_seed)
        seed_b = _seed_num(weak_seed)

        if prob_a is None or np.isnan(prob_a):
            if not np.isnan(seed_a) and not np.isnan(seed_b):
                winner_id = strong_id if seed_a < seed_b else weak_id
                winner_name = strong_name if winner_id == strong_id else weak_name
                winner_prob = 0.5
            else:
                winner_id = strong_id
                winner_name = strong_name
                winner_prob = 0.5
        else:
            if prob_a >= 0.5:
                winner_id = strong_id
                winner_name = strong_name
                winner_prob = prob_a
            else:
                winner_id = weak_id
                winner_name = weak_name
                winner_prob = 1 - prob_a

        winner_seed = seed_a if winner_id == strong_id else seed_b
        slot_winner[row["Slot"]] = {"TeamID": winner_id, "TeamName": winner_name, "Seed": winner_seed, "Prob": winner_prob}
        out_rows.append({
            **row,
            "StrongTeamID": strong_id,
            "StrongTeamName": strong_name,
            "WeakTeamID": weak_id,
            "WeakTeamName": weak_name,
            "WinnerTeamID": winner_id,
            "WinnerTeamName": winner_name,
            "WinnerSeed": winner_seed,
            "WinnerProb": winner_prob,
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(out_path, index=False)

    viz_lines = ["# 2026 NCAA M Bracket (Model Picks)", ""]
    for rnd, label in [
        (1.0, "Round 1"),
        (2.0, "Round 2"),
        (3.0, "Sweet 16"),
        (4.0, "Elite 8"),
        (5.0, "Final Four"),
        (6.0, "Championship"),
    ]:
        viz_lines.append(f"## {label}")
        subset = out[out["Slot"].str.startswith(f"R{int(rnd)}")]
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
