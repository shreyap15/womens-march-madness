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
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"


RAW_FILES = {
    "teams": "WTeams.csv",
    "conferences": "WTeamConferences.csv",
    "seeds": "WNCAATourneySeeds.csv",
    "regular_compact": "WRegularSeasonCompactResults.csv",
    "regular_detailed": "WRegularSeasonDetailedResults.csv",
    "tourney_compact": "WNCAATourneyCompactResults.csv",
    "tourney_detailed": "WNCAATourneyDetailedResults.csv",
    "tourney_slots": "WNCAATourneySlots.csv",
    "secondary_compact": "WSecondaryTourneyCompactResults.csv",
    "conference_tourney": "WConferenceTourneyGames.csv",
    "game_cities": "WGameCities.csv",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_raw(paths: DataPaths | None = None) -> Dict[str, pd.DataFrame]:
    paths = paths or DataPaths(project_root())
    data: Dict[str, pd.DataFrame] = {}
    for key, fname in RAW_FILES.items():
        fpath = paths.raw / fname
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
