"""Raw tables -> joined, audited player-match table."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .config import Paths, all_seasons
from .ingest import ingest_stats, ingest_votes
from .io_utils import write_json
from .join import JoinResult, check_join_rate, join_votes, load_aliases
from .manifest import build_manifest
from .qa import assert_qa, run_qa
from .store import sql_invariants, write_tables

STATS_FILE = "afltables_player_stats_{season}.parquet"
VOTES_FILE = "aflca_coaches_votes_{season}.parquet"


def default_aliases_path() -> Path:
    return Path(str(resources.files("ceb") / "resources" / "name_aliases.csv"))


def load_raw(raw_dir: Path, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing, st, vo = [], [], []
    for s in seasons:
        a, b = Path(raw_dir) / STATS_FILE.format(season=s), Path(raw_dir) / VOTES_FILE.format(season=s)
        if not a.exists() or not b.exists():
            missing.append(s)
            continue
        st.append(pd.read_parquet(a))
        vo.append(pd.read_parquet(b))
    if missing:
        raise FileNotFoundError(f"raw files missing for seasons {missing} in {raw_dir}: run `make data`")
    return pd.concat(st, ignore_index=True), pd.concat(vo, ignore_index=True)


def join_and_check(stats_raw: pd.DataFrame, votes_raw: pd.DataFrame, P: SimpleNamespace,
                   aliases_path: Path | None = None, include_test: bool = False,
                   enforce_rate: bool = True) -> tuple[JoinResult, dict, pd.DataFrame]:
    stats, s_info = ingest_stats(stats_raw)
    votes, v_info = ingest_votes(votes_raw)
    aliases = load_aliases(aliases_path or default_aliases_path())
    res = join_votes(stats, votes, P, aliases)
    res.report["ingest"] = {"stats": s_info, "votes": v_info}
    if enforce_rate:
        check_join_rate(res.report, P)
    qa = run_qa(res.player_match, P, include_test=include_test)
    return res, qa, stats


def _write(paths: Paths, res: JoinResult, qa: dict, P: SimpleNamespace, suffix: str) -> None:
    out = paths.processed
    out.mkdir(parents=True, exist_ok=True)
    res.player_match.to_parquet(out / f"player_match{suffix}.parquet", index=False)
    res.unmatched.to_csv(out / f"unmatched_votes{suffix}.csv", index=False)
    res.excluded.to_csv(out / f"excluded_matches{suffix}.csv", index=False)
    write_json(out / f"join_report{suffix}.json", res.report)
    write_json(out / f"qa_report{suffix}.json", qa)


def build(paths: Paths, P: SimpleNamespace, test: bool = False) -> dict:
    """``test=False``: Train+Validation seasons only. ``test=True``: the locked season (gated upstream)."""
    seasons = [P.splits.test_season] if test else all_seasons(P)
    stats_raw, votes_raw = load_raw(paths.raw, seasons)
    res, qa, _ = join_and_check(stats_raw, votes_raw, P, include_test=test)
    suffix = "_test" if test else ""
    _write(paths, res, qa, P, suffix)
    if not test:
        write_tables(paths.db, {"player_match": res.player_match, "excluded_matches": res.excluded,
                                "unmatched_votes": res.unmatched})
        sql = sql_invariants(paths.db, P.votes.total_per_match)
        write_json(paths.processed / "sql_invariants.json", sql)
        if sql["matches_vote_total_not_expected"] or sql["duplicate_player_matches"]:
            raise RuntimeError(f"SQL re-check disagrees with the join: {sql}")
        if paths.raw.exists():
            m = build_manifest(paths.raw, paths.manifest, seasons=all_seasons(P))
            write_json(paths.results / "data_manifest.json", m)
    else:
        m = build_manifest(paths.raw, paths.data / "manifest_test.json", seasons=[P.splits.test_season])
        write_json(paths.results / "data_manifest_test.json", m)
    assert_qa(qa)
    return {"join": res.report, "qa": qa}
