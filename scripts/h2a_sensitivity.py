"""How sensitive is the protocol's role-residual lens (H2a: defenders under-rated) to a planted premium?

SYNTHETIC data only. Plants a coaches' premium for defenders that no stat records, then asks whether
mean (actual - expected) votes for the 'Defender' role proxy comes out positive on validation seasons.
Output: results/synthetic_h2a_sensitivity.json  (reproduce with `uv run python scripts/h2a_sensitivity.py`)
"""

import copy
import sys
from pathlib import Path

from ceb import blindspots as BS
from ceb import evaluate as E
from ceb.config import _ns, load_params_dict
from ceb.io_utils import write_json
from ceb.pipeline import join_and_check
from ceb.splits import make_splits
from ceb.synth import Planted, generate

SCENARIOS = {
    "null: no planted effects": Planted(tackle_trend=0.0, defender_bonus=0.0, nonlinear=0.0),
    "defender premium 0.8 (comparable to the goals weight), otherwise linear":
        Planted(tackle_trend=0.0, defender_bonus=0.8, nonlinear=0.0),
    "defender premium 2.0 + planted non-linearity":
        Planted(tackle_trend=0.0, defender_bonus=2.0, nonlinear=1.5),
}


def params():
    d = copy.deepcopy(load_params_dict())
    d["splits"] = dict(train_start=2014, train_end=2019, val_seasons=[2020, 2021], test_season=2022,
                       latest_start_allowed=2016)
    d["m1"] = dict(alpha_grid=[0.001, 0.01], view_grid=["global", "match_z"])
    d["m2"].update(num_leaves_grid=[7, 15], min_data_in_leaf_grid=[50], rounds_grid=[50, 100, 200, 300])
    return _ns(d)


def main(out: Path):
    P, rows = params(), []
    for name, planted in SCENARIOS.items():
        sy = generate(list(range(2014, 2022)), 150, seed=11, planted=planted, include_finals=False)
        pm = join_and_check(sy.stats_raw, sy.votes_raw, P)[0].player_match
        sp = make_splits(pm, P)
        ch, _ = E.select_all(sp["train"], sp["val"], P)
        scored = E.fit_models(sp["train"], ch, P).score(sp["val"])
        for lens, col in (("M1 residual (protocol)", "ev_m1"), ("M2 residual (post-hoc)", "ev_m2")):
            h = BS.blind_spots(scored, P, "sens", ev_col=col)["h2a_defender"]
            rows.append({"scenario": name, "lens": lens, **{k: round(v, 3) for k, v in h.items() if k != "supported"},
                         "supported": h["supported"]})
            print(rows[-1], flush=True)
    write_json(out, {"synthetic": True, "note": "fictional data; shows sensitivity of the H2a lens, not AFL facts",
                     "rows": rows})


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/synthetic_h2a_sensitivity.json"))
