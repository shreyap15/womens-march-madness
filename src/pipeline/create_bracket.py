from __future__ import annotations

import pandas as pd

from src.data.load_data import load_raw


def create_bracket(
    season: int,
    out_path: str = "submissions/women/2026/bracket.csv",
    use_latest_if_missing: bool = True,
) -> None:
    raw = load_raw()
    slots = raw["tourney_slots"] if "tourney_slots" in raw else pd.read_csv("data/WNCAATourneySlots.csv")
    seeds = raw["seeds"]
    teams = raw["teams"][["TeamID", "TeamName"]]

    if use_latest_if_missing and season not in slots["Season"].unique():
        season = int(slots["Season"].max())
    slots = slots[slots["Season"] == season].copy()
    if use_latest_if_missing and season not in seeds["Season"].unique():
        season = int(seeds["Season"].max())
    seeds = seeds[seeds["Season"] == season].copy()

    if slots.empty:
        raise ValueError(f"No tournament slots found for season {season}.")
    if seeds.empty:
        raise ValueError(f"No tournament seeds found for season {season}.")

    seeds = seeds.merge(teams, on="TeamID", how="left")

    def seed_to_team(seed_code: str) -> pd.Series:
        row = seeds[seeds["Seed"] == seed_code]
        if row.empty:
            return pd.Series({"Seed": seed_code, "TeamID": None, "TeamName": None})
        r = row.iloc[0]
        return pd.Series({"Seed": r["Seed"], "TeamID": int(r["TeamID"]), "TeamName": r["TeamName"]})

    rows = []
    for _, s in slots.iterrows():
        strong = seed_to_team(s["StrongSeed"])
        weak = seed_to_team(s["WeakSeed"])
        rows.append(
            {
                "Season": season,
                "Slot": s["Slot"],
                "Round": int(str(s["Slot"])[1]) if str(s["Slot"]).startswith("R") else None,
                "StrongSeed": strong["Seed"],
                "StrongTeamID": strong["TeamID"],
                "StrongTeamName": strong["TeamName"],
                "WeakSeed": weak["Seed"],
                "WeakTeamID": weak["TeamID"],
                "WeakTeamName": weak["TeamName"],
            }
        )

    bracket = pd.DataFrame(rows).sort_values(["Round", "Slot"])
    bracket.to_csv(out_path, index=False)


if __name__ == "__main__":
    create_bracket(season=2026, out_path="submissions/women/2026/bracket_2026.csv")
