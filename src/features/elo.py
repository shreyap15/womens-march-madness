from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    initial: float = 1500.0
    k: float = 20.0
    regression_weight: float = 0.7  # 0.7 * old + 0.3 * initial


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _mov_multiplier(margin: float, r_winner: float, r_loser: float) -> float:
    return np.log(abs(margin) + 1.0) * (2.2 / ((r_winner - r_loser) * 0.001 + 2.2))


def _season_regress(rating: float, params: EloParams) -> float:
    return params.regression_weight * rating + (1.0 - params.regression_weight) * params.initial


def compute_elo(
    games: pd.DataFrame,
    season_col: str = "Season",
    wteam_col: str = "WTeamID",
    lteam_col: str = "LTeamID",
    wscore_col: str = "WScore",
    lscore_col: str = "LScore",
    params: EloParams | None = None,
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    Compute per-game Elo history.
    Returns (history_df, final_ratings).
    """
    params = params or EloParams()
    games = games.sort_values([season_col]).reset_index(drop=True)

    ratings: Dict[int, float] = {}
    history_rows = []
    current_season = None

    for _, row in games.iterrows():
        season = int(row[season_col])
        if current_season is None:
            current_season = season
        if season != current_season:
            # Offseason regression
            for team_id in list(ratings.keys()):
                ratings[team_id] = _season_regress(ratings[team_id], params)
            current_season = season

        wteam = int(row[wteam_col])
        lteam = int(row[lteam_col])
        wscore = float(row[wscore_col])
        lscore = float(row[lscore_col])
        margin = wscore - lscore

        rw = ratings.get(wteam, params.initial)
        rl = ratings.get(lteam, params.initial)

        ew = _expected(rw, rl)
        el = 1.0 - ew

        mov_mult = _mov_multiplier(margin, rw, rl)
        rw_new = rw + params.k * mov_mult * (1.0 - ew)
        rl_new = rl + params.k * mov_mult * (0.0 - el)

        history_rows.append(
            {
                "Season": season,
                "WTeamID": wteam,
                "LTeamID": lteam,
                "Elo_W_pre": rw,
                "Elo_L_pre": rl,
                "Elo_W_post": rw_new,
                "Elo_L_post": rl_new,
                "Margin": margin,
            }
        )

        ratings[wteam] = rw_new
        ratings[lteam] = rl_new

    history = pd.DataFrame(history_rows)
    return history, ratings
