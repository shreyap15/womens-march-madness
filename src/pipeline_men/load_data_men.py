from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / "men"

    @property
    def common(self) -> Path:
        return self.root / "data" / "raw" / "common"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"


RAW_FILES = {
    "teams": "MTeams.csv",
    "conferences": "MTeamConferences.csv",
    "seeds": "MNCAATourneySeeds.csv",
    "regular_compact": "MRegularSeasonCompactResults.csv",
    "regular_detailed": "MRegularSeasonDetailedResults.csv",
    "tourney_compact": "MNCAATourneyCompactResults.csv",
    "tourney_detailed": "MNCAATourneyDetailedResults.csv",
    "tourney_slots": "MNCAATourneySlots.csv",
    "secondary_compact": "MSecondaryTourneyCompactResults.csv",
    "conference_tourney": "MConferenceTourneyGames.csv",
    "game_cities": "MGameCities.csv",
    "team_spellings": "MTeamSpellings.csv",
    "conferences_master": "Conferences.csv",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_raw(paths: DataPaths | None = None) -> Dict[str, pd.DataFrame]:
    paths = paths or DataPaths(project_root())
    data: Dict[str, pd.DataFrame] = {}
    for key, fname in RAW_FILES.items():
        fpath = paths.common / fname if key == "conferences_master" else paths.raw / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing raw file: {fpath}")
        data[key] = pd.read_csv(fpath)
    return data


def load_processed(name: str, paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths(project_root())
    fpath = paths.processed / name
    if not fpath.exists():
        raise FileNotFoundError(f"Missing processed file: {fpath}")
    return pd.read_csv(fpath)
