from __future__ import annotations

from typing import Tuple

import pandas as pd


def _prep_compact(compact: pd.DataFrame) -> pd.DataFrame:
    w = compact.rename(
        columns={
            "WTeamID": "TeamID",
            "LTeamID": "OppTeamID",
            "WScore": "TeamScore",
            "LScore": "OppScore",
            "WLoc": "Loc",
        }
    )
    w["Win"] = 1

    l = compact.rename(
        columns={
            "LTeamID": "TeamID",
            "WTeamID": "OppTeamID",
            "LScore": "TeamScore",
            "WScore": "OppScore",
            "WLoc": "Loc",
        }
    )
    l["Win"] = 0
    l["Loc"] = l["Loc"].map({"H": "A", "A": "H", "N": "N"})

    return pd.concat([w, l], ignore_index=True)


def compute_long_history_features(compact: pd.DataFrame) -> pd.DataFrame:
    df = _prep_compact(compact)
    df["Margin"] = df["TeamScore"] - df["OppScore"]
    df["CloseGame"] = (df["Margin"].abs() <= 5).astype(int)

    grouped = df.groupby(["Season", "TeamID"], as_index=False).agg(
        Games=("Win", "count"),
        Wins=("Win", "sum"),
        MarginAvg=("Margin", "mean"),
        CloseGameWins=("Win", "sum"),
    )
    grouped["WinPct"] = grouped["Wins"] / grouped["Games"]

    close = df[df["CloseGame"] == 1].groupby(["Season", "TeamID"], as_index=False).agg(
        CloseGames=("Win", "count"),
        CloseWins=("Win", "sum"),
    )
    grouped = grouped.merge(close, on=["Season", "TeamID"], how="left")
    grouped["CloseWinPct"] = grouped["CloseWins"] / grouped["CloseGames"]

    loc = df.groupby(["Season", "TeamID", "Loc"], as_index=False).agg(
        LocGames=("Win", "count"),
        LocWins=("Win", "sum"),
    )
    loc["LocWinPct"] = loc["LocWins"] / loc["LocGames"]
    loc_pivot = loc.pivot_table(
        index=["Season", "TeamID"],
        columns="Loc",
        values="LocWinPct",
        aggfunc="first",
    ).reset_index()
    loc_pivot.columns.name = None
    loc_pivot = loc_pivot.rename(
        columns={"H": "HomeWinPct", "A": "AwayWinPct", "N": "NeutralWinPct"}
    )

    grouped = grouped.merge(loc_pivot, on=["Season", "TeamID"], how="left")
    return grouped


def compute_seed_features(seeds: pd.DataFrame) -> pd.DataFrame:
    df = seeds.copy()
    df["SeedNum"] = df["Seed"].str[1:3].astype(int)
    df["SeedPower"] = 1.0 / df["SeedNum"]
    return df[["Season", "TeamID", "SeedNum", "SeedPower"]]


def attach_conference_strength(
    team_features: pd.DataFrame,
    team_conferences: pd.DataFrame,
    elo_by_team: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute conference Elo by averaging team Elo within season and conference.
    """
    conf = team_conferences.rename(columns={"ConfAbbrev": "Conf"})
    df = team_features.merge(conf, on=["Season", "TeamID"], how="left")
    elo_conf = (
        elo_by_team.merge(conf, on=["Season", "TeamID"], how="left")
        .groupby(["Season", "Conf"], as_index=False)["Elo"]
        .mean()
        .rename(columns={"Elo": "ConferenceElo"})
    )
    df = df.merge(elo_conf, on=["Season", "Conf"], how="left")
    return df


def compute_conference_tourney_features(conference_tourney: pd.DataFrame) -> pd.DataFrame:
    df = conference_tourney.copy()
    df["Win"] = 1
    wins = df.groupby(["Season", "WTeamID"], as_index=False).agg(ConfTourneyWins=("Win", "sum"))
    wins = wins.rename(columns={"WTeamID": "TeamID"})
    return wins
