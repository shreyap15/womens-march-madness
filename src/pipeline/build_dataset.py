from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data.load_data import load_raw
from src.features.elo import compute_elo
from src.features.advanced_metrics import compute_efficiency_by_team
from src.features.team_stats import (
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
    # Last post-elo per season/team
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

    last10 = (
        all_games.sort_values(["Season", "TeamID", "GameIndex"])
        .groupby(["Season", "TeamID"], as_index=False)
        .tail(1)[["Season", "TeamID", "Last10NetRtg"]]
    )
    return last10


def build_team_features(raw: dict) -> pd.DataFrame:
    elo_by_team = build_elo_by_team(raw)
    long_features = compute_long_history_features(raw["regular_compact"])
    seeds = compute_seed_features(raw["seeds"])
    conf_tourney = compute_conference_tourney_features(raw["conference_tourney"])

    # Advanced efficiency (2010+)
    eff = compute_efficiency_by_team(raw["regular_detailed"])
    recent = _recent_form(raw["regular_detailed"])
    eff = eff.merge(recent, on=["Season", "TeamID"], how="left")
    eff["AdjNetRtg"] = 0.7 * eff["NetRtg"] + 0.3 * eff["Last10NetRtg"]

    team = long_features.merge(elo_by_team, on=["Season", "TeamID"], how="left")
    team = team.merge(eff, on=["Season", "TeamID"], how="left")
    team = team.merge(seeds, on=["Season", "TeamID"], how="left")
    team = team.merge(conf_tourney, on=["Season", "TeamID"], how="left")
    team = attach_conference_strength(team, raw["conferences"], elo_by_team)

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

    # Feature list pulled from team features
    feature_cols = [
        "Elo",
        "AdjNetRtg",
        "NetRtg",
        "eFG",
        "TS",
        "OREB_rate",
        "DREB_rate",
        "TO_rate",
        "AST_TO",
        "SeedNum",
        "SeedPower",
        "ConferenceElo",
        "WinPct",
        "MarginAvg",
        "CloseWinPct",
        "HomeWinPct",
        "AwayWinPct",
        "NeutralWinPct",
    ]

    tourney = raw["tourney_detailed"]
    rows: List[dict] = []
    for _, game in tourney.iterrows():
        rows.extend(_matchup_rows(game, team, feature_cols))

    dataset = pd.DataFrame(rows)
    diff_cols = [f"{c}_diff" for c in feature_cols]
    return dataset, diff_cols


def main() -> None:
    raw = load_raw()
    dataset, diff_cols = build_training_dataset(raw)
    dataset.to_csv("data/processed/training_dataset.csv", index=False)
    pd.Series(diff_cols).to_csv("data/processed/feature_cols.txt", index=False, header=False)


if __name__ == "__main__":
    main()
