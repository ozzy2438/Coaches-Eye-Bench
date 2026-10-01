"""Data manifest: source, fetch time, row counts and SHA-256 per raw file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import read_json, sha256_file, utc_now, write_json

_SOURCES = {
    "afltables_player_stats": "AFL Tables season-scoped pages via fitzRoy::scrape_afltables_match",
    "aflca_coaches_votes": "AFLCA Champion Player of the Year leaderboard via fitzRoy (scrape_coaches_votes)",
    "squiggle_games": "Squiggle API https://api.squiggle.com.au (games)",
}


def _season_of(name: str) -> int | None:
    stem = name.rsplit(".", 1)[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def build_manifest(raw_dir: Path, out: Path, seasons: list[int] | None = None) -> dict:
    """Hash raw files. ``seasons`` restricts the manifest so dev steps never open the test season."""
    raw_dir = Path(raw_dir)
    meta_path = raw_dir / "fetch_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    files = []
    for p in sorted(raw_dir.glob("*")):
        if p.name == "fetch_meta.json" or p.is_dir():
            continue
        if seasons is not None and _season_of(p.name) not in seasons:
            continue
        stem = p.name.rsplit("_", 1)[0]
        rows = int(len(pd.read_parquet(p))) if p.suffix == ".parquet" else None
        files.append({"file": p.name, "source": _SOURCES.get(stem, "unknown"), "rows": rows,
                      "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    manifest = {"created_utc": utc_now(), "fetch_meta": meta, "files": files}
    write_json(out, manifest)
    return manifest
