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


def _build_h2h_map(regular: pd.DataFrame, season: int) -> dict:
    reg = regular[regular["Season"] == season]
    h2h = {}
    for _, g in reg.iterrows():
        w = int(g["WTeamID"])
        l = int(g["LTeamID"])
        margin = float(g["WScore"] - g["LScore"])
        key = (min(w, l), max(w, l))
        if key not in h2h:
            h2h[key] = {"games": 0, "wins_a": 0, "margin_a": 0.0}
        rec = h2h[key]
        rec["games"] += 1
        if w == key[0]:
            rec["wins_a"] += 1
            rec["margin_a"] += margin
        else:
            rec["margin_a"] -= margin
    return h2h


def _build_matchups(
    team_features: pd.DataFrame,
    season: int,
    base_cols: list[str],
    h2h_map: dict,
    upset_rates: dict,
) -> pd.DataFrame:
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
        key = (min(a, b), max(a, b))
        rec = h2h_map.get(key, {"games": 0, "wins_a": 0, "margin_a": 0.0})
        games = rec["games"]
        wins_a = rec["wins_a"]
        margin_a = rec["margin_a"] / games if games > 0 else np.nan
        winpct_a = wins_a / games if games > 0 else np.nan
        is_a_small = a == key[0]
        h2h_win = winpct_a if is_a_small else (1 - winpct_a if games > 0 else np.nan)
        h2h_margin = margin_a if is_a_small else (-margin_a if games > 0 else np.nan)
        row["H2HGames"] = games
        row["H2HWinPct"] = h2h_win
        row["H2HMargin"] = h2h_margin

        # Seed upset interactions and matchup upset rate
        seed_a = ta.get("SeedNum", np.nan)
        seed_b = tb.get("SeedNum", np.nan)
        if pd.notna(seed_a) and pd.notna(seed_b):
            low = min(seed_a, seed_b)
            high = max(seed_a, seed_b)
            row["SeedMatchupUpsetRate"] = upset_rates.get((low, high), np.nan)
        else:
            row["SeedMatchupUpsetRate"] = np.nan
        sd = abs(row.get("SeedNum_diff", np.nan)) if "SeedNum_diff" in row else np.nan
        row["ClassicUpsetSeed"] = 1 if pd.notna(sd) and sd in [3, 4, 5, 6, 7] else 0
        if "NetRtg_diff" in row and "SeedNum_diff" in row:
            row["NetRtgVsSeed"] = row["NetRtg_diff"] * (-row["SeedNum_diff"])
        else:
            row["NetRtgVsSeed"] = np.nan
        if "ESPNRank_diff" in row and "SeedNum_diff" in row:
            row["ESPNSeedResidualDiff"] = row["ESPNRank_diff"] - row["SeedNum_diff"]
            row["SeedResidualInteraction"] = row["SeedNum_diff"] * row["ESPNSeedResidualDiff"]
        else:
            row["ESPNSeedResidualDiff"] = np.nan
            row["SeedResidualInteraction"] = np.nan
        if "FTERank_diff" in row and "SeedNum_diff" in row:
            row["FTESeedResidualDiff"] = row["FTERank_diff"] - row["SeedNum_diff"]
            row["FTESeedResidualInteraction"] = row["SeedNum_diff"] * row["FTESeedResidualDiff"]
        else:
            row["FTESeedResidualDiff"] = np.nan
            row["FTESeedResidualInteraction"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _missing_report(X: pd.DataFrame, out_path: str) -> pd.DataFrame:
    missing = X.isna().mean().sort_values(ascending=False)
    report = missing.reset_index()
    report.columns = ["feature", "missing_pct"]
    report.to_csv(out_path, index=False)
    return report


def generate_predictions(
    season: int,
    out_path: str = "submissions/submission.csv",
    strict_missing_check: bool = True,
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
    # Ensure Elo is defined for the Elo-only signal
    if "Elo" in team_features.columns:
        team_features["Elo"] = team_features["Elo"].fillna(1500.0)

    bundle = joblib.load("models/saved_models.pkl")
    feature_cols = bundle["feature_cols"]
    diff_cols = [c for c in feature_cols if c.endswith("_diff")]
    base_cols = [c.replace("_diff", "") for c in diff_cols]
    h2h_map = _build_h2h_map(raw["regular_compact"], season)
    # upset rates from tournament history
    seed_map = raw["seeds"].copy()
    seed_map["SeedNum"] = seed_map["Seed"].str[1:3].astype(int)
    seed_lookup = seed_map[["Season", "TeamID", "SeedNum"]]
    tourney = raw["tourney_compact"]
    tourney_seed = tourney.merge(
        seed_lookup.rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}),
        on=["Season", "WTeamID"],
        how="left",
    ).merge(
        seed_lookup.rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}),
        on=["Season", "LTeamID"],
        how="left",
    )
    tourney_seed["SeedHigh"] = tourney_seed[["WSeed", "LSeed"]].max(axis=1)
    tourney_seed["SeedLow"] = tourney_seed[["WSeed", "LSeed"]].min(axis=1)
    tourney_seed["Upset"] = (tourney_seed["WSeed"] > tourney_seed["LSeed"]).astype(int)
    upset_rates_df = (
        tourney_seed.groupby(["SeedLow", "SeedHigh"], as_index=False)["Upset"]
        .mean()
        .rename(columns={"Upset": "SeedMatchupUpsetRate"})
    )
    upset_rates = {
        (int(r["SeedLow"]), int(r["SeedHigh"])): float(r["SeedMatchupUpsetRate"])
        for _, r in upset_rates_df.iterrows()
    }

    matchup = _build_matchups(
        team_features,
        season,
        base_cols,
        h2h_map,
        upset_rates,
    )
    X = matchup[feature_cols]

    # Missing value diagnostics for prediction-time feature coverage
    report = _missing_report(X, "data/processed/feature_missing_2026.csv")
    core_features = [
        "Elo_diff",
        "AdjNetRtg_diff",
        "NetRtg_diff",
        "eFG_diff",
        "TS_diff",
        "WinPct_diff",
        "MarginAvg_diff",
    ]
    core_missing = report[report["feature"].isin(core_features)]
    if strict_missing_check and not core_missing.empty:
        bad = core_missing[core_missing["missing_pct"] > 0.10]
        if not bad.empty:
            raise ValueError(
                "High missing rate for core features: "
                + ", ".join(f"{r.feature}={r.missing_pct:.2%}" for r in bad.itertuples())
            )

    log_p = bundle["logistic"].predict_proba(X)[:, 1]
    rf_p = bundle["rf_pipeline"].predict_proba(X)[:, 1]
    preds = {
        "logistic": bundle["cal_log"].transform(log_p),
        "rf": bundle["cal_rf"].transform(rf_p),
    }

    # Use calibrated logistic regression as primary submission model
    calibrated = preds["logistic"]

    ids = matchup.apply(lambda r: f"{season}_{int(r['TeamA'])}_{int(r['TeamB'])}", axis=1)
    submission = pd.DataFrame({"ID": ids, "Pred": calibrated})
    submission.to_csv(out_path, index=False)

    # Write winner/loser list (based on 0.5 threshold) as primary pairs file
    winners = np.where(
        calibrated >= 0.5,
        matchup["TeamA"].astype(int).values,
        matchup["TeamB"].astype(int).values,
    )
    losers = np.where(
        calibrated >= 0.5,
        matchup["TeamB"].astype(int).values,
        matchup["TeamA"].astype(int).values,
    )
    winners_out = pd.DataFrame({"WTeamID": winners, "LTeamID": losers})
    winners_out.to_csv("submissions/WNCAATourneyPredictions.csv", index=False)

    # Write predictions separately
    preds_out = pd.DataFrame(
        {
            "ID": ids,
            "Pred": calibrated,
        }
    )
    preds_out.to_csv("submissions/WNCAATourneyPredictions_with_preds.csv", index=False)


if __name__ == "__main__":
    generate_predictions(season=2026, strict_missing_check=False)
