from __future__ import annotations

import numpy as np
import pandas as pd


def compute_long_history_features(regular: pd.DataFrame) -> pd.DataFrame:
    df = regular.copy()
    df["Margin"] = df["WScore"] - df["LScore"]

    w = df[["Season", "WTeamID", "LTeamID", "Margin", "WLoc"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"}
    )
    l = df[["Season", "LTeamID", "WTeamID", "Margin", "WLoc"]].rename(
        columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"}
    )
    l["Margin"] = -l["Margin"]
    l["WLoc"] = l["WLoc"].map({"H": "A", "A": "H", "N": "N"})

    games = pd.concat([w, l], ignore_index=True)
    games["Win"] = (games["Margin"] > 0).astype(int)
    games["Close"] = (games["Margin"].abs() <= 5).astype(int)
    games["CloseWin"] = ((games["Margin"] > 0) & (games["Close"] == 1)).astype(int)

    summary = (
        games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            WinPct=("Win", "mean"),
            MarginAvg=("Margin", "mean"),
            MarginStd=("Margin", "std"),
            CloseWinPct=("CloseWin", "mean"),
        )
    )

    # Home/Away/Neutral win rates
    for loc, col in [("H", "HomeWinPct"), ("A", "AwayWinPct"), ("N", "NeutralWinPct")]:
        loc_df = games[games["WLoc"] == loc]
        loc_sum = loc_df.groupby(["Season", "TeamID"], as_index=False)["Win"].mean()
        loc_sum = loc_sum.rename(columns={"Win": col})
        summary = summary.merge(loc_sum, on=["Season", "TeamID"], how="left")

    return summary


def compute_seed_features(seeds: pd.DataFrame) -> pd.DataFrame:
    df = seeds.copy()
    df["SeedNum"] = df["Seed"].str[1:3].astype(int)
    df["SeedPower"] = 1.0 / df["SeedNum"]
    return df[["Season", "TeamID", "Seed", "SeedNum", "SeedPower"]]


def compute_conference_tourney_features(conf_tourney: pd.DataFrame) -> pd.DataFrame:
    df = conf_tourney.copy()
    wins = df.groupby(["Season", "WTeamID"], as_index=False).size().rename(columns={"size": "ConfTourneyWins"})
    wins = wins.rename(columns={"WTeamID": "TeamID"})
    return wins


def attach_conference_strength(team: pd.DataFrame, confs: pd.DataFrame, elo_by_team: pd.DataFrame) -> pd.DataFrame:
    conf = confs.copy()
    elo = elo_by_team.rename(columns={"TeamID": "TeamID", "Elo": "TeamElo"})
    merged = conf.merge(elo, on=["Season", "TeamID"], how="left")
    conf_elo = merged.groupby(["Season", "ConfAbbrev"], as_index=False)["TeamElo"].mean()
    conf_elo = conf_elo.rename(columns={"TeamElo": "ConferenceElo"})
    team = team.merge(conf, on=["Season", "TeamID"], how="left")
    team = team.merge(conf_elo, on=["Season", "ConfAbbrev"], how="left")
    return team
