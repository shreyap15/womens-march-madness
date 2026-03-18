from __future__ import annotations

import math
import pandas as pd


def _expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def compute_elo(games: pd.DataFrame, season_col: str = "Season") -> tuple[pd.DataFrame, dict[int, float]]:
    ratings = {}
    history = []

    for _, g in games.sort_values([season_col, "DayNum"]).iterrows():
        season = int(g[season_col])
        w = int(g["WTeamID"])
        l = int(g["LTeamID"])
        margin = float(g["WScore"] - g["LScore"])

        # offseason regression
        if season not in ratings:
            ratings = {k: 0.7 * v + 0.3 * 1500 for k, v in ratings.items()}

        r_w = ratings.get(w, 1500.0)
        r_l = ratings.get(l, 1500.0)

        expected_w = _expected_score(r_w, r_l)
        actual_w = 1.0
        mov_mult = math.log(abs(margin) + 1) * (2.2 / ((r_w - r_l) * 0.001 + 2.2))
        k = 20.0
        delta = k * mov_mult * (actual_w - expected_w)

        r_w_post = r_w + delta
        r_l_post = r_l - delta

        ratings[w] = r_w_post
        ratings[l] = r_l_post

        history.append(
            {
                season_col: season,
                "WTeamID": w,
                "LTeamID": l,
                "Elo_W_pre": r_w,
                "Elo_L_pre": r_l,
                "Elo_W_post": r_w_post,
                "Elo_L_post": r_l_post,
                "Margin": margin,
            }
        )

    return pd.DataFrame(history), ratings
