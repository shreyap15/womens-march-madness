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
    # Compute recent Elo trajectory and season peak from history
    history = history.copy()
    history["DayNum"] = raw["regular_compact"].merge(
        history[["Season", "WTeamID", "LTeamID", "Margin"]],
        left_on=["Season", "WTeamID", "LTeamID"],
        right_on=["Season", "WTeamID", "LTeamID"],
        how="right",
    )["DayNum"]
    history = history.sort_values(["Season", "DayNum"])
    w_hist = history[["Season", "DayNum", "WTeamID", "Elo_W_post"]].rename(
        columns={"WTeamID": "TeamID", "Elo_W_post": "EloPost"}
    )
    l_hist = history[["Season", "DayNum", "LTeamID", "Elo_L_post"]].rename(
        columns={"LTeamID": "TeamID", "Elo_L_post": "EloPost"}
    )
    all_hist = pd.concat([w_hist, l_hist], ignore_index=True).sort_values(["Season", "DayNum"])
    # Use last 30 days of regular season (DayNum >= 100) as proxy window
    last_30 = all_hist[all_hist["DayNum"] >= 100]
    elo_traj = last_30.groupby(["Season", "TeamID"], as_index=False).agg(
        EloRecent=("EloPost", "last"),
        EloEarly=("EloPost", "first"),
    )
    elo_traj["EloTrajectory"] = elo_traj["EloRecent"] - elo_traj["EloEarly"]
    elo_peak = all_hist.groupby(["Season", "TeamID"], as_index=False)["EloPost"].max().rename(
        columns={"EloPost": "EloPeak"}
    )
    elo_by_team = elo_by_team.merge(elo_traj[["Season", "TeamID", "EloTrajectory"]], on=["Season", "TeamID"], how="left")
    elo_by_team = elo_by_team.merge(elo_peak, on=["Season", "TeamID"], how="left")
    elo_by_team["EloCurrentVsPeak"] = elo_by_team["Elo"] / elo_by_team["EloPeak"]
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

    # Tournament program performance history (regular season net vs tourney net)
    tourney_eff = compute_efficiency_by_team(raw["tourney_detailed"])
    tourney_eff = tourney_eff.rename(columns={"NetRtg": "TourneyNetRtg"})
    reg_eff = eff[["Season", "TeamID", "NetRtg"]].rename(columns={"NetRtg": "RegNetRtg"})
    perf = tourney_eff.merge(reg_eff, on=["Season", "TeamID"], how="left")
    perf["TourneyDelta"] = perf["TourneyNetRtg"] - perf["RegNetRtg"]
    perf = perf.sort_values(["TeamID", "Season"])
    perf["ProgramTourneyEff"] = (
        perf.groupby("TeamID")["TourneyDelta"].shift(1).expanding().mean().reset_index(level=0, drop=True)
    )
    perf = perf[["Season", "TeamID", "ProgramTourneyEff"]]

    team = long_features.merge(elo_by_team, on=["Season", "TeamID"], how="left")
    team = team.merge(eff, on=["Season", "TeamID"], how="left")
    team = team.merge(perf, on=["Season", "TeamID"], how="left")
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

    # ESPN ratings (additional signal from external polls)
    try:
        espn = pd.read_csv("data/espn_ratings/wncaa_espn.csv")
        espn = espn.rename(columns={"Votes": "ESPNVotes"})
        espn["ESPNRank"] = espn.groupby("Season")["ESPNVotes"].rank(ascending=False, method="min")
        team = team.merge(espn[["Season", "TeamID", "ESPNVotes"]], on=["Season", "TeamID"], how="left")
        team = team.merge(espn[["Season", "TeamID", "ESPNRank"]], on=["Season", "TeamID"], how="left")
    except FileNotFoundError:
        pass

    # FiveThirtyEight ratings (additional external signal)
    try:
        fte = pd.read_csv("data/fivethirtyeight_ratings/538ratingsWomen.csv")
        fte = fte.rename(columns={"538rating": "FTERating"})
        fte["FTERank"] = fte.groupby("Season")["FTERating"].rank(ascending=False, method="min")
        team = team.merge(fte[["Season", "TeamID", "FTERating"]], on=["Season", "TeamID"], how="left")
        team = team.merge(fte[["Season", "TeamID", "FTERank"]], on=["Season", "TeamID"], how="left")
    except FileNotFoundError:
        pass

    # Coaching stability proxy: rolling win% std over last 3 seasons (shifted)
    stability = team[["Season", "TeamID", "WinPct"]].copy().sort_values(["TeamID", "Season"])
    stability["WinPctStd3"] = (
        stability.groupby("TeamID")["WinPct"]
        .rolling(3, min_periods=1)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    team = team.merge(stability[["Season", "TeamID", "WinPctStd3"]], on=["Season", "TeamID"], how="left")

    # Quality of wins: average opponent net rating in wins, weighted by margin
    reg_compact = raw["regular_compact"].copy()
    reg_compact["Margin"] = reg_compact["WScore"] - reg_compact["LScore"]
    opp_net = eff[["Season", "TeamID", "NetRtg"]].rename(columns={"TeamID": "LTeamID", "NetRtg": "OppNetRtg"})
    wins = reg_compact.merge(opp_net, on=["Season", "LTeamID"], how="left")
    wins["Weight"] = wins["Margin"].clip(lower=1)
    qow = (
        wins.groupby(["Season", "WTeamID"])
        .apply(lambda g: (g["OppNetRtg"] * g["Weight"]).sum() / g["Weight"].sum())
        .rename("QualityWins")
        .reset_index()
        .rename(columns={"WTeamID": "TeamID"})
    )
    team = team.merge(qow, on=["Season", "TeamID"], how="left")

    # Coach features (2026 snapshot, if available)
    try:
        coaches = pd.read_csv("data/WNCAAWCoaches_2026_mapped.csv")
        coaches = coaches.rename(
            columns={
                "TenureYears_2026": "CoachTenureYears",
                "IsVacant": "CoachIsVacant",
                "IsInterim": "CoachIsInterim",
            }
        )
        coaches["Season"] = 2026
        team = team.merge(
            coaches[["Season", "TeamID", "CoachTenureYears", "CoachIsVacant", "CoachIsInterim"]],
            on=["Season", "TeamID"],
            how="left",
        )
    except FileNotFoundError:
        pass

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
        "RoadPerformanceGap",
        "QualityWins",
        "SeedNum",
        "SeedPower",
        "ConferenceElo",
        "OppElo",
        "OppNetRtg",
        "ConfTourneyWins",
        "ProgramTourneyEff",
        "WinPct",
        "MarginAvg",
        "MarginStd",
        "CloseWinPct",
        "HomeWinPct",
        "AwayWinPct",
        "NeutralWinPct",
        "ESPNRank",
        "ESPNVotes",
        "FTERank",
        "FTERating",
        "WinPctStd3",
        "CoachTenureYears",
        "CoachIsVacant",
        "CoachIsInterim",
    ]

    tourney = raw["tourney_compact"]
    # Historical upset rate by seed matchup (women's)
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

    # Head-to-head features from regular season
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
        # For row_w (TeamA=a), if a is the smaller id we use wins_a else invert
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

        # Seed upset interactions (classic upset bands and seed quality vs net rating)
        for row in matchup_rows:
            seed_a = row.get("SeedNum_diff", np.nan)
            # Seed difference as absolute for matchup type
            sd = abs(seed_a) if not np.isnan(seed_a) else np.nan
            row["ClassicUpsetSeed"] = 1 if sd in [3, 4, 5, 6, 7] else 0
            # Historical upset rate for this seed matchup
            seed_a_val = team_idx.loc[(season, row["TeamA"])].get("SeedNum", np.nan) if (season, row["TeamA"]) in team_idx.index else np.nan
            seed_b_val = team_idx.loc[(season, row["TeamB"])].get("SeedNum", np.nan) if (season, row["TeamB"]) in team_idx.index else np.nan
            if pd.notna(seed_a_val) and pd.notna(seed_b_val):
                low = min(seed_a_val, seed_b_val)
                high = max(seed_a_val, seed_b_val)
                match = upset_rates[(upset_rates["SeedLow"] == low) & (upset_rates["SeedHigh"] == high)]
                row["SeedMatchupUpsetRate"] = float(match["SeedMatchupUpsetRate"].iloc[0]) if not match.empty else np.nan
            else:
                row["SeedMatchupUpsetRate"] = np.nan
            if "NetRtg_diff" in row and "SeedNum_diff" in row:
                # TeamA net rating relative to seed expectation
                row["NetRtgVsSeed"] = row["NetRtg_diff"] * (-row["SeedNum_diff"])
            else:
                row["NetRtgVsSeed"] = np.nan
            # ESPN seed residual and interaction
            if "ESPNRank_diff" in row and "SeedNum_diff" in row:
                row["ESPNSeedResidualDiff"] = row["ESPNRank_diff"] - row["SeedNum_diff"]
                row["SeedResidualInteraction"] = row["SeedNum_diff"] * row["ESPNSeedResidualDiff"]
            else:
                row["ESPNSeedResidualDiff"] = np.nan
                row["SeedResidualInteraction"] = np.nan
            # 538 seed residual and interaction
            if "FTERank_diff" in row and "SeedNum_diff" in row:
                row["FTESeedResidualDiff"] = row["FTERank_diff"] - row["SeedNum_diff"]
                row["FTESeedResidualInteraction"] = row["SeedNum_diff"] * row["FTESeedResidualDiff"]
            else:
                row["FTESeedResidualDiff"] = np.nan
                row["FTESeedResidualInteraction"] = np.nan
        rows.extend(matchup_rows)

    dataset = pd.DataFrame(rows)
    diff_cols = [f"{c}_diff" for c in feature_cols]
    extra_cols = [
        "H2HGames",
        "H2HWinPct",
        "H2HMargin",
        "ClassicUpsetSeed",
        "SeedMatchupUpsetRate",
        "NetRtgVsSeed",
        "ESPNSeedResidualDiff",
        "SeedResidualInteraction",
        "FTESeedResidualDiff",
        "FTESeedResidualInteraction",
    ]
    return dataset, diff_cols + extra_cols


def main() -> None:
    raw = load_raw()
    dataset, diff_cols = build_training_dataset(raw)
    dataset.to_csv("data/processed/women/training_dataset.csv", index=False)
    pd.Series(diff_cols).to_csv("data/processed/women/feature_cols.txt", index=False, header=False)


if __name__ == "__main__":
    main()
