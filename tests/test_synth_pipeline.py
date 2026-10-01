import duckdb
import pandas as pd

from ceb.config import Paths
from ceb.io_utils import read_json, sha256_file
from ceb.pipeline import STATS_FILE, VOTES_FILE, build
from ceb.synth import generate


def test_synth_structure_and_determinism():
    a = generate([2022, 2023], 30, seed=5)
    b = generate([2022, 2023], 30, seed=5)
    c = generate([2022, 2023], 30, seed=6)
    pd.testing.assert_frame_equal(a.stats_raw, b.stats_raw)
    assert not a.stats_raw.equals(c.stats_raw)
    v = a.votes_raw.assign(v=a.votes_raw["Coaches.Votes"].astype(int))
    assert (v.groupby(["Season", "Round", "Home.Team"])["v"].sum() == 30).all()
    ha = a.stats_raw[a.stats_raw["Round"] != "QF"]
    assert (ha.groupby(["Season", "Round", "Home.team"]).size() == 44).all()
    assert (ha.groupby(["Season", "Round", "Home.team"])["Brownlow.Votes"].sum() == 6).all()
    assert (a.stats_raw["Round"] == "QF").any()  # finals present so the filter is exercised
    names = a.votes_raw["Player.Name"]
    assert names.str.contains(r"\(").any() and names.str.contains(",").any()  # perturbations present


def test_flawed_matches_are_generated_on_request():
    f = generate([2022], 40, seed=1, flaw_rate=0.25)
    v = f.votes_raw.assign(v=f.votes_raw["Coaches.Votes"].astype(int))
    tot = v.groupby(["Season", "Round", "Home.Team"])["v"].sum()
    assert (tot != 30).any()


def test_raw_parquet_to_processed_roundtrip(tmp_path, SP):
    paths = Paths(tmp_path / "data", tmp_path / "results").ensure()
    sy = generate([2022, 2023, 2024, 2025], 36, seed=9)
    for s in (2022, 2023, 2024, 2025):
        sy.stats_raw[sy.stats_raw["Season"] == s].to_parquet(paths.raw / STATS_FILE.format(season=s))
        sy.votes_raw[sy.votes_raw["Season"] == s].to_parquet(paths.raw / VOTES_FILE.format(season=s))
    out = build(paths, SP)
    assert out["join"]["overall"]["match_rate_rows"] == 1.0 and out["qa"]["ok"]
    dev = pd.read_parquet(paths.processed / "player_match.parquet")
    assert SP.splits.test_season not in set(dev["season"])  # the locked season never reaches dev data
    sql = read_json(paths.processed / "sql_invariants.json")
    assert sql["matches_vote_total_not_expected"] == 0 and sql["duplicate_player_matches"] == 0
    assert sql["matches"] == dev["match_id"].nunique() and sql["player_matches"] == len(dev)
    con = duckdb.connect(str(paths.db), read_only=True)
    assert con.execute("select count(*) from player_match").fetchone()[0] == len(dev)
    con.close()
    man = read_json(paths.manifest)
    assert len(man["files"]) == 6  # 3 dev seasons x 2 sources; the locked season is never opened
    f = man["files"][0]
    assert f["sha256"] == sha256_file(paths.raw / f["file"]) and f["rows"] > 0
    assert (paths.results / "data_manifest.json").exists()
    t = build(paths, SP, test=True)
    assert t["join"]["overall"]["match_rate_rows"] == 1.0
    test_pm = pd.read_parquet(paths.processed / "player_match_test.parquet")
    assert set(test_pm["season"]) == {SP.splits.test_season}
    assert len(read_json(paths.data / "manifest_test.json")["files"]) == 2
