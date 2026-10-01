"""Season-scoped builds, audit files on gate failure, raw-cache override and test-season gating."""

import pandas as pd
import pytest

from ceb import cli
from ceb.config import Paths
from ceb.io_utils import read_json
from ceb.join import JoinRateError
from ceb.pipeline import STATS_FILE, VOTES_FILE, build
from ceb.synth import generate


def _write_raw(raw, seasons, corrupt_names=False):
    sy = generate(seasons, 30, seed=4)
    v = sy.votes_raw
    if corrupt_names:  # unknown names push the join below the 99% floor
        v = v.assign(**{"Player.Name": v["Player.Name"].where(v.index % 10 != 0, "Nobody Atall")})
    for s in seasons:
        sy.stats_raw[sy.stats_raw["Season"] == s].to_parquet(raw / STATS_FILE.format(season=s))
        v[v["Season"] == s].to_parquet(raw / VOTES_FILE.format(season=s))


@pytest.fixture
def paths(tmp_path):
    return Paths(tmp_path / "data", tmp_path / "results").ensure()


def test_season_subset_builds_only_those_seasons(paths, SP):
    _write_raw(paths.raw, [2022, 2023, 2024])
    build(paths, SP, seasons=[2024, 2023])
    pm = pd.read_parquet(paths.processed / "player_match.parquet")
    assert set(pm["season"]) == {2023, 2024}
    assert {f["file"] for f in read_json(paths.manifest)["files"]} == {
        STATS_FILE.format(season=s) for s in (2023, 2024)} | {VOTES_FILE.format(season=s) for s in (2023, 2024)}


@pytest.mark.parametrize("kw", [dict(seasons=[]), dict(seasons=[2025]), dict(seasons=[2019]),
                                dict(test=True, seasons=[2024])])
def test_invalid_season_requests_are_refused(paths, SP, kw):
    # 2025 is the smoke test season, 2019 is outside the splits; custom seasons never apply to test builds
    with pytest.raises(ValueError):
        build(paths, SP, **kw)


def test_audit_files_survive_a_failed_join_gate(paths, SP):
    _write_raw(paths.raw, [2022, 2023, 2024], corrupt_names=True)
    with pytest.raises(JoinRateError):
        build(paths, SP)
    unmatched = pd.read_csv(paths.processed / "unmatched_votes.csv")
    assert len(unmatched) > 0 and (unmatched["name_raw"] == "Nobody Atall").all()
    assert (paths.processed / "join_report.json").exists()


def test_raw_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CEB_RAW_DIR", str(tmp_path / "shared_raw"))
    assert Paths(tmp_path / "pilot").raw == tmp_path / "shared_raw"
    monkeypatch.delenv("CEB_RAW_DIR")
    assert Paths(tmp_path / "pilot").raw == tmp_path / "pilot" / "raw"


@pytest.mark.parametrize("argv", [["guard", "--test"], ["build", "--test"]])
def test_test_season_commands_refuse_without_frozen_choices(tmp_path, monkeypatch, argv):
    monkeypatch.setenv("CEB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CEB_RESULTS_DIR", str(tmp_path / "results"))
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert "frozen_choices" in str(e.value)
