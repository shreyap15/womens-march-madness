from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline_men.load_data_men import load_raw
from src.features_men.advanced_metrics_men import compute_efficiency_by_team


def predict_championship_total(
    season: int, out_path: str = "submissions/men/2026/championship_total.csv"
) -> None:
    raw = load_raw()
    detailed = raw["regular_detailed"]
    detailed = detailed[detailed["Season"] == season].copy()
    team_features = compute_efficiency_by_team(detailed)

    # If bracket results exist, use final matchup
    try:
        bracket = pd.read_csv("submissions/men/2026/bracket_2026_results.csv")
        final = bracket[bracket["Slot"] == "R6CH"]
        if not final.empty:
            final = final.iloc[0]
            team_a = int(final["StrongTeamID"])
            team_b = int(final["WeakTeamID"])
        else:
            team_a = None
            team_b = None
    except FileNotFoundError:
        team_a = None
        team_b = None

    if team_a is None or team_b is None:
        return

    team_features = team_features.set_index("TeamID")
    ta = team_features.loc[team_a]
    tb = team_features.loc[team_b]

    poss = np.nanmean([ta.get("PossPerGame"), tb.get("PossPerGame")])
    off_a = ta.get("OffRtg")
    def_a = ta.get("DefRtg")
    off_b = tb.get("OffRtg")
    def_b = tb.get("DefRtg")

    points_a = poss * np.nanmean([off_a, def_b])
    points_b = poss * np.nanmean([off_b, def_a])
    expected_total = float(points_a + points_b)

    out = pd.DataFrame([
        {
            "Season": season,
            "FinalSlot": "R6CH",
            "ExpectedCombinedScore": round(expected_total, 2),
            "MostLikelyTeamA": team_a,
            "MostLikelyTeamB": team_b,
            "MostLikelyPairTotal": round(expected_total, 2),
        }
    ])
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    predict_championship_total(season=2026)
