from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.pipeline_men.load_data_men import load_raw
from src.features_men.elo_men import compute_elo
from src.features_men.advanced_metrics_men import compute_efficiency_by_team
from src.features_men.team_stats_men import (
    compute_conference_tourney_features,
    compute_long_history_features,
    compute_seed_features,
    attach_conference_strength,
)


def _combine_compact(regular: pd.DataFrame, tourney: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    cols = ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]
    combined = pd.concat(
        [
            regular[cols],
            tourney[cols],
            secondary[cols],
        ],
        ignore_index=True,
    ).sort_values(["Season", "DayNum"])
    return combined


def build_elo_by_team(raw: dict) -> pd.DataFrame:
    combined = _combine_compact(
        raw["regular_compact"], raw["tourney_compact"], raw["secondary_compact"]
    )
    history, _ = compute_elo(combined, season_col="Season")

    winners = history[["Season", "WTeamID", "Elo_W_post"]].rename(
        columns={"WTeamID": "TeamID", "Elo_W_post": "Elo"}
    )
    losers = history[["Season", "LTeamID", "Elo_L_post"]].rename(
        columns={"LTeamID": "TeamID", "Elo_L_post": "Elo"}
    )
    merged = pd.concat([winners, losers], ignore_index=True)
    elo_by_team = (
        merged.sort_values(["Season"])
        .groupby(["Season", "TeamID"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return elo_by_team


def _recent_form(detailed: pd.DataFrame) -> pd.DataFrame:
    df = detailed.copy().sort_values(["Season", "DayNum"])
    w = df.assign(
        TeamID=df["WTeamID"],
        OppID=df["LTeamID"],
        Points=df["WScore"],
        OppPoints=df["LScore"],
        FGA=df["WFGA"],
        OREB=df["WOR"],
        TO=df["WTO"],
        FTA=df["WFTA"],
    )
    l = df.assign(
        TeamID=df["LTeamID"],
        OppID=df["WTeamID"],
        Points=df["LScore"],
        OppPoints=df["WScore"],
        FGA=df["LFGA"],
        OREB=df["LOR"],
        TO=df["LTO"],
        FTA=df["LFTA"],
    )
    all_games = pd.concat([w, l], ignore_index=True)

    def poss(row: pd.Series) -> float:
        return row["FGA"] - row["OREB"] + row["TO"] + 0.475 * row["FTA"]

    all_games["Poss"] = all_games.apply(poss, axis=1)
    all_games["OffRtg"] = all_games["Points"] / all_games["Poss"]
    all_games["DefRtg"] = all_games["OppPoints"] / all_games["Poss"]
    all_games["NetRtg"] = all_games["OffRtg"] - all_games["DefRtg"]

    all_games["GameIndex"] = all_games.groupby(["Season", "TeamID"]).cumcount()
    all_games["Last10NetRtg"] = (
        all_games.groupby(["Season", "TeamID"])["NetRtg"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    all_games["Last5NetRtg"] = (
        all_games.groupby(["Season", "TeamID"])["NetRtg"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    all_games["TrendNetRtg"] = all_games["Last5NetRtg"] - all_games["Last10NetRtg"]

    last10 = (
        all_games.sort_values(["Season", "TeamID", "GameIndex"])
        .groupby(["Season", "TeamID"], as_index=False)
        .tail(1)[["Season", "TeamID", "Last10NetRtg", "Last5NetRtg", "TrendNetRtg"]]
    )
    return last10


def _load_kenpom_2026() -> pd.DataFrame:
    try:
        kp = pd.read_csv("data/processed/men/kenpom_men_2026.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    # normalize team names for merge
    kp["TeamNameClean"] = kp["Team"].str.lower().str.replace("&", "and", regex=False)
    kp["TeamNameClean"] = kp["TeamNameClean"].str.replace("[^a-z0-9 ]", "", regex=True).str.strip()
    return kp


def build_team_features(raw: dict) -> pd.DataFrame:
    elo_by_team = build_elo_by_team(raw)
    long_features = compute_long_history_features(raw["regular_compact"])
    seeds = compute_seed_features(raw["seeds"])
    conf_tourney = compute_conference_tourney_features(raw["conference_tourney"])

    eff = compute_efficiency_by_team(raw["regular_detailed"])
    recent = _recent_form(raw["regular_detailed"])
    eff = eff.merge(recent, on=["Season", "TeamID"], how="left")
    eff["AdjNetRtg"] = 0.7 * eff["NetRtg"] + 0.3 * eff["Last10NetRtg"]

    team = long_features.merge(elo_by_team, on=["Season", "TeamID"], how="left")
    team = team.merge(eff, on=["Season", "TeamID"], how="left")
    team = team.merge(seeds, on=["Season", "TeamID"], how="left")
    team = team.merge(conf_tourney, on=["Season", "TeamID"], how="left")
    team = attach_conference_strength(team, raw["conferences"], elo_by_team)

    # Strength of schedule
    regular = raw["regular_compact"]
    w = regular[["Season", "WTeamID", "LTeamID"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"}
    )
    l = regular[["Season", "WTeamID", "LTeamID"]].rename(
        columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"}
    )
    games = pd.concat([w, l], ignore_index=True)
    opp_elo = elo_by_team.rename(columns={"TeamID": "OppTeamID", "Elo": "OppElo"})
    opp_net = eff[["Season", "TeamID", "NetRtg"]].rename(
        columns={"TeamID": "OppTeamID", "NetRtg": "OppNetRtg"}
    )
    sos = games.merge(opp_elo, on=["Season", "OppTeamID"], how="left").merge(
        opp_net, on=["Season", "OppTeamID"], how="left"
    )
    sos = (
        sos.groupby(["Season", "TeamID"], as_index=False)
        .agg(OppElo=("OppElo", "mean"), OppNetRtg=("OppNetRtg", "mean"))
    )
    team = team.merge(sos, on=["Season", "TeamID"], how="left")

    # KenPom features (2026 snapshot)
    kp = _load_kenpom_2026()
    if not kp.empty:
        spell = raw["team_spellings"].copy()
        spell["TeamNameClean"] = spell["TeamNameSpelling"].str.lower().str.replace("&", "and", regex=False)
        spell["TeamNameClean"] = spell["TeamNameClean"].str.replace("[^a-z0-9 ]", "", regex=True).str.strip()
        kp = kp.merge(spell[["TeamNameClean", "TeamID"]], on="TeamNameClean", how="left")
        kp["Season"] = 2026
        team = team.merge(
            kp[[
                "Season",
                "TeamID",
                "NetRtg",
                "ORtg",
                "DRtg",
                "AdjT",
                "Luck",
                "Trapezoid",
                "Odds",
                "Other",
                "Injuries",
            ]].rename(
                columns={
                    "NetRtg": "KenPomNetRtg",
                    "ORtg": "KenPomORtg",
                    "DRtg": "KenPomDRtg",
                    "AdjT": "KenPomAdjT",
                    "Luck": "KenPomLuck",
                    "Trapezoid": "KenPomTrapezoid",
                    "Odds": "KenPomOdds",
                }
            ),
            on=["Season", "TeamID"],
            how="left",
        )

    return team


def _matchup_rows(game_row: pd.Series, team_features: pd.DataFrame, feature_cols: List[str]) -> List[dict]:
    season = int(game_row["Season"])
    w = int(game_row["WTeamID"])
    l = int(game_row["LTeamID"])

    t = team_features.set_index(["Season", "TeamID"])
    default = pd.Series({c: np.nan for c in feature_cols})
    tw = t.loc[(season, w)][feature_cols] if (season, w) in t.index else default
    tl = t.loc[(season, l)][feature_cols] if (season, l) in t.index else default

    def diff(a: pd.Series, b: pd.Series) -> dict:
        out = {}
        for col in feature_cols:
            out[f"{col}_diff"] = a[col] - b[col]
        return out

    row_w = {
        "Season": season,
        "TeamA": w,
        "TeamB": l,
        "Target": 1,
        **diff(tw, tl),
    }
    row_l = {
        "Season": season,
        "TeamA": l,
        "TeamB": w,
        "Target": 0,
        **diff(tl, tw),
    }
    return [row_w, row_l]


def build_training_dataset(raw: dict) -> Tuple[pd.DataFrame, List[str]]:
    team = build_team_features(raw)

    feature_cols = [
        "Elo",
        "AdjNetRtg",
        "Last5NetRtg",
        "Last10NetRtg",
        "TrendNetRtg",
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
        "SeedNum",
        "SeedPower",
        "ConferenceElo",
        "OppElo",
        "OppNetRtg",
        "ConfTourneyWins",
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
    ]

    tourney = raw["tourney_compact"]
    seed_map = raw["seeds"].copy()
    seed_map["SeedNum"] = seed_map["Seed"].str[1:3].astype(int)
    seed_lookup = seed_map[["Season", "TeamID", "SeedNum"]]
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
    upset_rates = (
        tourney_seed.groupby(["SeedLow", "SeedHigh"], as_index=False)["Upset"]
        .mean()
        .rename(columns={"Upset": "SeedMatchupUpsetRate"})
    )
    rows: List[dict] = []
    team_idx = team.set_index(["Season", "TeamID"])

    reg = raw["regular_compact"]
    h2h = {}
    for _, g in reg.iterrows():
        season = int(g["Season"])
        w = int(g["WTeamID"])
        l = int(g["LTeamID"])
        margin = float(g["WScore"] - g["LScore"])
        key = (season, min(w, l), max(w, l))
        if key not in h2h:
            h2h[key] = {"games": 0, "wins_a": 0, "margin_a": 0.0}
        rec = h2h[key]
        rec["games"] += 1
        if w == key[1]:
            rec["wins_a"] += 1
            rec["margin_a"] += margin
        else:
            rec["margin_a"] -= margin

    for _, game in tourney.iterrows():
        matchup_rows = _matchup_rows(game, team, feature_cols)
        season = int(game["Season"])
        a = int(game["WTeamID"])
        b = int(game["LTeamID"])
        key = (season, min(a, b), max(a, b))
        rec = h2h.get(key, {"games": 0, "wins_a": 0, "margin_a": 0.0})
        games = rec["games"]
        wins_a = rec["wins_a"]
        margin_a = rec["margin_a"] / games if games > 0 else np.nan
        winpct_a = wins_a / games if games > 0 else np.nan
        is_a_small = a == key[1]
        h2h_win = winpct_a if is_a_small else (1 - winpct_a if games > 0 else np.nan)
        h2h_margin = margin_a if is_a_small else (-margin_a if games > 0 else np.nan)
        for row in matchup_rows:
            if row["TeamA"] == a:
                row["H2HGames"] = games
                row["H2HWinPct"] = h2h_win
                row["H2HMargin"] = h2h_margin
            else:
                row["H2HGames"] = games
                row["H2HWinPct"] = (1 - h2h_win) if games > 0 else np.nan
                row["H2HMargin"] = -h2h_margin if games > 0 else np.nan

        for row in matchup_rows:
            seed_a_val = team_idx.loc[(season, row["TeamA"])].get("SeedNum", np.nan) if (season, row["TeamA"]) in team_idx.index else np.nan
            seed_b_val = team_idx.loc[(season, row["TeamB"])].get("SeedNum", np.nan) if (season, row["TeamB"]) in team_idx.index else np.nan
            if pd.notna(seed_a_val) and pd.notna(seed_b_val):
                low = min(seed_a_val, seed_b_val)
                high = max(seed_a_val, seed_b_val)
                match = upset_rates[(upset_rates["SeedLow"] == low) & (upset_rates["SeedHigh"] == high)]
                row["SeedMatchupUpsetRate"] = float(match["SeedMatchupUpsetRate"].iloc[0]) if not match.empty else np.nan
            else:
                row["SeedMatchupUpsetRate"] = np.nan
        rows.extend(matchup_rows)

    dataset = pd.DataFrame(rows)
    diff_cols = [f"{c}_diff" for c in feature_cols]
    extra_cols = ["H2HGames", "H2HWinPct", "H2HMargin", "SeedMatchupUpsetRate"]
    return dataset, diff_cols + extra_cols


def main() -> None:
    raw = load_raw()
    dataset, diff_cols = build_training_dataset(raw)
    dataset.to_csv("data/processed/men/training_dataset_men.csv", index=False)
    pd.Series(diff_cols).to_csv("data/processed/men/feature_cols_men.txt", index=False, header=False)


if __name__ == "__main__":
    main()
