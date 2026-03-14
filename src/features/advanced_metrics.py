from __future__ import annotations

import pandas as pd


def _possessions(fga: pd.Series, oreb: pd.Series, to: pd.Series, fta: pd.Series) -> pd.Series:
    return fga - oreb + to + 0.475 * fta


def compute_efficiency_by_team(detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team per-season efficiency metrics from detailed results.
    Expects the women's detailed results schema.
    """
    df = detailed.copy()

    # Winner side
    w_poss = _possessions(df["WFGA"], df["WOR"], df["WTO"], df["WFTA"])
    l_poss = _possessions(df["LFGA"], df["LOR"], df["LTO"], df["LFTA"])

    w = pd.DataFrame(
        {
            "Season": df["Season"],
            "TeamID": df["WTeamID"],
            "Points": df["WScore"],
            "OppPoints": df["LScore"],
            "FGA": df["WFGA"],
            "FGM": df["WFGM"],
            "FGM3": df["WFGM3"],
            "FTA": df["WFTA"],
            "OREB": df["WOR"],
            "DREB": df["WDR"],
            "TO": df["WTO"],
            "AST": df["WAst"],
            "Poss": w_poss,
            "OppPoss": l_poss,
        }
    )

    l = pd.DataFrame(
        {
            "Season": df["Season"],
            "TeamID": df["LTeamID"],
            "Points": df["LScore"],
            "OppPoints": df["WScore"],
            "FGA": df["LFGA"],
            "FGM": df["LFGM"],
            "FGM3": df["LFGM3"],
            "FTA": df["LFTA"],
            "OREB": df["LOR"],
            "DREB": df["LDR"],
            "TO": df["LTO"],
            "AST": df["LAst"],
            "Poss": l_poss,
            "OppPoss": w_poss,
        }
    )

    all_games = pd.concat([w, l], ignore_index=True)
    all_games["OffRtg"] = all_games["Points"] / all_games["Poss"]
    all_games["DefRtg"] = all_games["OppPoints"] / all_games["Poss"]
    all_games["NetRtg"] = all_games["OffRtg"] - all_games["DefRtg"]

    grouped = all_games.groupby(["Season", "TeamID"], as_index=False).sum()
    games = all_games.groupby(["Season", "TeamID"], as_index=False).size().rename(columns={"size": "Games"})
    grouped = grouped.merge(games, on=["Season", "TeamID"], how="left")
    net_std = (
        all_games.groupby(["Season", "TeamID"], as_index=False)["NetRtg"]
        .std()
        .rename(columns={"NetRtg": "NetRtgStd"})
    )
    grouped = grouped.merge(net_std, on=["Season", "TeamID"], how="left")

    grouped["OffRtg"] = grouped["Points"] / grouped["Poss"]
    grouped["DefRtg"] = grouped["OppPoints"] / grouped["OppPoss"]
    grouped["NetRtg"] = grouped["OffRtg"] - grouped["DefRtg"]
    grouped["eFG"] = (grouped["FGM"] + 0.5 * grouped["FGM3"]) / grouped["FGA"]
    grouped["TS"] = grouped["Points"] / (2 * (grouped["FGA"] + 0.44 * grouped["FTA"]))

    grouped["OREB_rate"] = grouped["OREB"] / (grouped["OREB"] + grouped["DREB"])
    grouped["DREB_rate"] = grouped["DREB"] / (grouped["OREB"] + grouped["DREB"])
    grouped["TO_rate"] = grouped["TO"] / grouped["Poss"]
    grouped["AST_TO"] = grouped["AST"] / grouped["TO"].replace(0, 1)
    grouped["PossPerGame"] = grouped["Poss"] / grouped["Games"]
    grouped = grouped.rename(columns={"Games": "GamesPlayed"})
    grouped["NetRtgConf"] = grouped["NetRtg"] / grouped["NetRtgStd"].replace(0, pd.NA)

    # Home/Away splits for road performance gap
    home_games = df[df["WLoc"] == "H"]
    away_games = df[df["WLoc"] == "A"]

    def _team_split(split_df: pd.DataFrame, side: str) -> pd.DataFrame:
        if side == "home":
            w = split_df.assign(
                TeamID=split_df["WTeamID"],
                Points=split_df["WScore"],
                OppPoints=split_df["LScore"],
                FGA=split_df["WFGA"],
                OREB=split_df["WOR"],
                TO=split_df["WTO"],
                FTA=split_df["WFTA"],
            )
            l = split_df.assign(
                TeamID=split_df["LTeamID"],
                Points=split_df["LScore"],
                OppPoints=split_df["WScore"],
                FGA=split_df["LFGA"],
                OREB=split_df["LOR"],
                TO=split_df["LTO"],
                FTA=split_df["LFTA"],
            )
        else:
            w = split_df.assign(
                TeamID=split_df["WTeamID"],
                Points=split_df["WScore"],
                OppPoints=split_df["LScore"],
                FGA=split_df["WFGA"],
                OREB=split_df["WOR"],
                TO=split_df["WTO"],
                FTA=split_df["WFTA"],
            )
            l = split_df.assign(
                TeamID=split_df["LTeamID"],
                Points=split_df["LScore"],
                OppPoints=split_df["WScore"],
                FGA=split_df["LFGA"],
                OREB=split_df["LOR"],
                TO=split_df["LTO"],
                FTA=split_df["LFTA"],
            )
        all_split = pd.concat([w, l], ignore_index=True)
        all_split["Poss"] = _possessions(all_split["FGA"], all_split["OREB"], all_split["TO"], all_split["FTA"])
        all_split["OffRtg"] = all_split["Points"] / all_split["Poss"]
        all_split["DefRtg"] = all_split["OppPoints"] / all_split["Poss"]
        return all_split.groupby(["Season", "TeamID"], as_index=False).agg(
            OffRtg=("OffRtg", "mean"),
            DefRtg=("DefRtg", "mean"),
        )

    home = _team_split(home_games, "home").rename(
        columns={"OffRtg": "HomeOffRtg", "DefRtg": "HomeDefRtg"}
    )
    away = _team_split(away_games, "away").rename(
        columns={"OffRtg": "AwayOffRtg", "DefRtg": "AwayDefRtg"}
    )

    grouped = grouped.merge(home, on=["Season", "TeamID"], how="left")
    grouped = grouped.merge(away, on=["Season", "TeamID"], how="left")
    grouped["RoadNetRtg"] = grouped["AwayOffRtg"] - grouped["AwayDefRtg"]
    grouped["HomeNetRtg"] = grouped["HomeOffRtg"] - grouped["HomeDefRtg"]
    grouped["RoadPerformanceGap"] = grouped["RoadNetRtg"] - grouped["HomeNetRtg"]

    return grouped
