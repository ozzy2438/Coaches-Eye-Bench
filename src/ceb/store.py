"""DuckDB as the local analytical store, and an independent SQL re-check of the key invariants."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def write_tables(db_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for name, df in tables.items():
            con.register("_t", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
            con.unregister("_t")
    finally:
        con.close()


def sql_invariants(db_path: Path, total_votes: int) -> dict:
    """Second, independent implementation of the checks that matter most (SQL, not pandas)."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
        return {
            "matches": q("SELECT COUNT(DISTINCT match_id) FROM player_match"),
            "player_matches": q("SELECT COUNT(*) FROM player_match"),
            "matches_vote_total_not_expected": q(
                f"SELECT COUNT(*) FROM (SELECT match_id FROM player_match GROUP BY match_id "
                f"HAVING SUM(votes) <> {total_votes})"),
            "duplicate_player_matches": q(
                "SELECT COUNT(*) FROM (SELECT match_id, player_key FROM player_match "
                "GROUP BY match_id, player_key HAVING COUNT(*) > 1)"),
            "negative_stat_rows": q(
                "SELECT COUNT(*) FROM player_match WHERE kicks < 0 OR handballs < 0 OR goals < 0 "
                "OR tackles < 0 OR marks < 0"),
        }
    finally:
        con.close()
