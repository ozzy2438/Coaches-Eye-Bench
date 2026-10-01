"""Protocol parameters (read from the frozen protocol.md) and filesystem layout."""

from __future__ import annotations

import copy
import hashlib
import os
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "protocol.md"
_BLOCK = re.compile(r"```toml protocol-params\n(.*?)```", re.S)


def _ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def load_params_dict(path: Path = PROTOCOL_PATH) -> dict:
    m = _BLOCK.search(Path(path).read_text(encoding="utf-8"))
    if m is None:
        raise ValueError(f"no `toml protocol-params` block in {path}")
    return tomllib.loads(m.group(1))


def load_params(path: Path = PROTOCOL_PATH) -> SimpleNamespace:
    """The protocol parameters as nested attributes, e.g. ``P.splits.test_season``."""
    return _ns(load_params_dict(path))


def protocol_sha256(path: Path = PROTOCOL_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke_params() -> SimpleNamespace:
    """Tiny synthetic configuration for ``make smoke`` and tests.

    Same code paths as the real protocol, different (fake) seasons and small grids.
    It is never used for anything reported as a result.
    """
    d = copy.deepcopy(load_params_dict())
    d["splits"] = {
        "train_start": 2022,
        "train_end": 2023,
        "val_seasons": [2024],
        "test_season": 2025,
        "latest_start_allowed": 2023,
    }
    d["inference"].update(bootstrap_B=200, permutation_n=2000)
    d["m1"] = {"alpha_grid": [0.001, 0.1], "view_grid": ["global", "match_z"]}
    d["m2"].update(
        num_leaves_grid=[7],
        min_data_in_leaf_grid=[50],
        rounds_grid=[25, 50, 100],
    )
    d["q2"].update(bootstrap_B=200, brownlow_bootstrap_B=40)
    d["trends"].update(bootstrap_B=40, block_width=1, last_season_included=2024)
    d["coverage"].update(min_test_valid_match_share=0.9)
    return _ns(d)


def all_seasons(P: SimpleNamespace, include_test: bool = False) -> list[int]:
    s = list(range(P.splits.train_start, P.splits.train_end + 1)) + list(P.splits.val_seasons)
    if include_test:
        s.append(P.splits.test_season)
    return s


class Paths:
    """Where things live. Override with CEB_DATA_DIR / CEB_RESULTS_DIR."""

    def __init__(self, data: Path | None = None, results: Path | None = None):
        self.data = Path(data or os.environ.get("CEB_DATA_DIR", ROOT / "data"))
        self.results = Path(results or os.environ.get("CEB_RESULTS_DIR", ROOT / "results"))

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def models(self) -> Path:
        return self.data / "models"

    @property
    def manifest(self) -> Path:
        return self.data / "manifest.json"

    @property
    def db(self) -> Path:
        return self.data / "ceb.duckdb"

    def ensure(self) -> Paths:
        for p in (self.raw, self.processed, self.models, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self
