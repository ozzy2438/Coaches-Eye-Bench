"""Role proxy from box scores (descriptive heuristic; not validated against true positions)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

ROLES = ["Ruck", "Midfielder", "Forward", "Defender", "Other", "Unclassified"]


def assign_roles(pm: pd.DataFrame, P: SimpleNamespace) -> pd.Series:
    """Per player-season from per-game averages, in priority order Ruck > Midfielder > Forward >
    Defender > Other; fewer than ``min_games_for_role`` games -> Unclassified."""
    r = P.roles
    g = pm.groupby(["season", "player_key"])
    m = g[["hit_outs", "clearances", "goals", "marks_inside_50", "rebounds", "one_percenters"]].mean()
    m["games"] = g.size()
    role = np.select(
        [
            m["games"] < P.q2.min_games_for_role,
            m["hit_outs"] >= r.ruck_hit_outs,
            m["clearances"] >= r.midfielder_clearances,
            (m["goals"] >= r.forward_goals) | (m["marks_inside_50"] >= r.forward_marks_inside_50),
            (m["rebounds"] >= r.defender_rebounds) | (m["one_percenters"] >= r.defender_one_percenters),
        ],
        ["Unclassified", "Ruck", "Midfielder", "Forward", "Defender"],
        default="Other",
    )
    lut = pd.Series(role, index=m.index, name="role")
    keys = pd.MultiIndex.from_frame(pm[["season", "player_key"]])
    return pd.Series(lut.reindex(keys).to_numpy(), index=pm.index, name="role")
