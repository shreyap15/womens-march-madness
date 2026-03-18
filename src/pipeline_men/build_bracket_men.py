from __future__ import annotations

import re
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.pipeline_men.load_data_men import load_raw


FIELD_2026 = {
    "East": {
        1: "Duke",
        16: "Siena",
        8: "Ohio State",
        9: "TCU",
        5: "St. John's",
        12: "Northern Iowa",
        4: "Kansas",
        13: "Cal Baptist",
        6: "Louisville",
        11: "South Florida",
        3: "Michigan State",
        14: "North Dakota State",
        7: "UCLA",
        10: "UCF",
        2: "UConn",
        15: "Furman",
    },
    "Midwest": {
        1: "Michigan",
        16: "Howard",
        8: "Georgia",
        9: "Saint Louis",
        5: "Texas Tech",
        12: "Akron",
        4: "Alabama",
        13: "Hofstra",
        6: "Tennessee",
        11: ["Miami (OH)", "SMU"],
        3: "Virginia",
        14: "Wright State",
        7: "Kentucky",
        10: "Santa Clara",
        2: "Iowa State",
        15: "Tennessee State",
    },
    "South": {
        1: "Florida",
        16: ["Prairie View A&M", "Lehigh"],
        8: "Clemson",
        9: "Iowa",
        5: "Vanderbilt",
        12: "McNeese",
        4: "Nebraska",
        13: "Troy",
        6: "North Carolina",
        11: "VCU",
        3: "Illinois",
        14: "Penn",
        7: "Saint Mary's",
        10: "Texas A&M",
        2: "Houston",
        15: "Idaho",
    },
    "West": {
        1: "Arizona",
        16: "Long Island (LIU)",
        8: "Villanova",
        9: "Utah State",
        5: "Wisconsin",
        12: "High Point",
        4: "Arkansas",
        13: "Hawaii",
        6: "BYU",
        11: "Texas",
        3: "Gonzaga",
        14: "Kennesaw State",
        7: "Miami (FL)",
        10: "Missouri",
        2: "Purdue",
        15: "Queens",
    },
}

REGION_TO_LETTER = {
    "East": "W",
    "Midwest": "X",
    "South": "Y",
    "West": "Z",
}

NAME_ALIASES = {
    "ohio state": "Ohio St",
    "st johns": "St John's",
    "st. johns": "St John's",
    "michigan state": "Michigan St",
    "north dakota state": "N Dakota St",
    "uconn": "Connecticut",
    "saint louis": "St Louis",
    "wright state": "Wright St",
    "iowa state": "Iowa St",
    "tennessee state": "Tennessee St",
    "prairie view a&m": "Prairie View",
    "prairie view aandm": "Prairie View",
    "saint marys": "St Mary's CA",
    "saint mary's": "St Mary's CA",
    "texas a&m": "Texas A&M",
    "long island (liu)": "LIU Brooklyn",
    "long island": "LIU Brooklyn",
    "liu": "LIU Brooklyn",
    "utah state": "Utah St",
    "kennesaw state": "Kennesaw",
    "miami (oh)": "Miami OH",
    "miami (fl)": "Miami FL",
    "hawaii": "Hawaii",
    "mcneese": "McNeese St",
    "queens": "Queens NC",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _build_name_map(teams: pd.DataFrame) -> Dict[str, int]:
    name_to_id = {}
    for _, row in teams.iterrows():
        name_to_id[_norm(str(row["TeamName"]))] = int(row["TeamID"])
    for raw, alias in NAME_ALIASES.items():
        norm_alias = _norm(alias)
        norm_raw = _norm(raw)
        if norm_alias in name_to_id:
            name_to_id[norm_raw] = name_to_id[norm_alias]
    return name_to_id


def _resolve_team_id(name: str, name_to_id: Dict[str, int]) -> int | None:
    key = _norm(name)
    return name_to_id.get(key)


def _choose_playin(team_names: Iterable[str], name_to_id: Dict[str, int], pred_map: Dict[str, float]) -> str:
    teams = list(team_names)
    if len(teams) != 2:
        return teams[0]
    a, b = teams
    a_id = _resolve_team_id(a, name_to_id)
    b_id = _resolve_team_id(b, name_to_id)
    if a_id is None or b_id is None:
        return a
    lo, hi = (a_id, b_id) if a_id < b_id else (b_id, a_id)
    key = f"2026_{lo}_{hi}"
    p = pred_map.get(key)
    if p is None:
        return a
    p_a = p if a_id == lo else 1 - p
    return a if p_a >= 0.5 else b


def build_bracket_2026(
    out_path: str = "submissions/men/2026/bracket_2026.csv",
    preds_path: str = "submissions/men/2026/MNCAATourneyPredictions_with_preds.csv",
    use_manual_field: bool = True,
) -> None:
    raw = load_raw()
    seeds = raw["seeds"]
    slots = raw["tourney_slots"]

    season = 2026
    if season not in seeds["Season"].unique():
        season = int(seeds["Season"].max())
    if season not in slots["Season"].unique():
        season = int(slots["Season"].max())

    seeds = seeds[seeds["Season"] == season].copy()
    slots = slots[slots["Season"] == season].copy()

    teams = raw["teams"][["TeamID", "TeamName"]].copy()
    name_to_id = _build_name_map(teams)

    pred_map = {}
    if use_manual_field:
        try:
            preds = pd.read_csv(preds_path)
            pred_map = dict(zip(preds["ID"], preds["Pred"]))
        except FileNotFoundError:
            pred_map = {}

    if use_manual_field:
        seed_to_team = {}
        for region, seeds_in_region in FIELD_2026.items():
            letter = REGION_TO_LETTER.get(region)
            if letter is None:
                continue
            for seed_num, team_name in seeds_in_region.items():
                if isinstance(team_name, list):
                    team_name = _choose_playin(team_name, name_to_id, pred_map)
                team_id = _resolve_team_id(str(team_name), name_to_id)
                if team_id is None:
                    raise ValueError(f"Could not resolve team name '{team_name}' for {region} seed {seed_num}.")
                seed_key = f"{letter}{seed_num:02d}"
                seed_to_team[seed_key] = team_id
    else:
        seed_to_team = dict(zip(seeds["Seed"], seeds["TeamID"]))

    # create rows with TeamID and TeamName for round 1 and play-ins
    teams = raw["teams"][['TeamID','TeamName']].set_index('TeamID')

    rows = []
    for _, row in slots.iterrows():
        strong = row["StrongSeed"]
        weak = row["WeakSeed"]
        slot = row["Slot"]
        if isinstance(slot, str) and slot.startswith("R"):
            try:
                round_num = float(int(slot[1]))
            except ValueError:
                round_num = np.nan
        else:
            round_num = np.nan
        strong_id = seed_to_team.get(strong, np.nan)
        weak_id = seed_to_team.get(weak, np.nan)
        strong_name = teams.at[strong_id, 'TeamName'] if strong_id in teams.index else np.nan
        weak_name = teams.at[weak_id, 'TeamName'] if weak_id in teams.index else np.nan
        rows.append({
            "Season": season,
            "Slot": slot,
            "Round": round_num,
            "StrongSeed": strong,
            "StrongTeamID": strong_id,
            "StrongTeamName": strong_name,
            "WeakSeed": weak,
            "WeakTeamID": weak_id,
            "WeakTeamName": weak_name,
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    build_bracket_2026()
