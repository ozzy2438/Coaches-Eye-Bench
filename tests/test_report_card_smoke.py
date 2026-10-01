import json
import re

import pandas as pd
import pytest

from ceb import report
from ceb.card import build_card
from ceb.smoke import run_smoke
from ceb.squiggle import cross_check, games_frame, user_agent


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    run_smoke(out / "s", seed=11, matches_per_season=40, log=lambda *_: None)
    return out / "s"


def test_smoke_outputs_are_stamped_synthetic(smoke_dir):
    for name in ("val/metrics.json", "test/metrics.json", "val/q2_summary.json", "test/test_run.json",
                 "val/frozen_choices.json"):
        d = json.loads((smoke_dir / name).read_text())
        assert d["stamp"]["synthetic"] if "stamp" in d else d["synthetic"], name
    assert (smoke_dir / "SYNTHETIC.json").exists()
    for f in ("ndcg5", "calibration"):
        assert (smoke_dir / "val" / "figures" / f"{f}.png").stat().st_size > 5000
    assert {"q3_trends.csv", "q3_separation.csv", "q2_role.csv", "q2_posthoc_m2_role.csv",
            "q2_coaches_vs_umpires.csv"} <= {p.name for p in (smoke_dir / "val").iterdir()}


def test_report_banners_synthetic_and_readme_block_is_replaced(smoke_dir, tmp_path):
    md, block = report.build_results(smoke_dir)
    assert "SYNTHETIC FIXTURE OUTPUT - NOT A RESULT" in md and "Test:" in md
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\nOLD\n{report.END}\noutro\n")
    report.write_report(smoke_dir, readme)
    t = readme.read_text()
    assert "OLD" not in t and t.startswith("intro") and t.rstrip().endswith("outro")


def test_pending_block_when_no_real_results(tmp_path):
    (tmp_path / "val").mkdir()
    md, block = report.build_results(tmp_path)
    assert "real-data run pending" in block and "No number in this repository is a finding" in block
    assert not re.search(r"NDCG@5 [0-9]", block)  # no invented numbers


def test_match_review_card(smoke_dir, tmp_path):
    scores = pd.read_parquet(smoke_dir / "derived" / "test_scores.parquet")
    rnd = int(scores["round"].iloc[0])
    out = build_card(scores, int(scores["season"].iloc[0]), rnd, tmp_path / "card.html", synthetic=True)
    h = out.read_text()
    assert "SYNTHETIC DATA" in h and "Not affiliated with any club or the AFL" in h and "<table>" in h
    with pytest.raises(ValueError):
        build_card(scores, 1999, 1, tmp_path / "x.html")


def test_squiggle_parsing_crosscheck_and_contact_requirement(monkeypatch):
    games = [
        {"year": 2024, "round": 1, "hteam": "Greater Western Sydney", "ateam": "Geelong", "hscore": 100,
         "ascore": 90, "is_final": 0},
        {"year": 2024, "round": 1, "hteam": "Collingwood", "ateam": "Carlton", "hscore": 80, "ascore": 70,
         "is_final": 0},
        {"year": 2024, "round": 25, "hteam": "Collingwood", "ateam": "Carlton", "hscore": 1, "ascore": 2,
         "is_final": 1},
    ]
    g = games_frame(games)
    assert len(g) == 2 and set(g["home_team"]) == {"GWS", "Collingwood"}
    stats = pd.DataFrame({"match_id": ["a", "b"], "season": 2024, "round": 1,
                          "home_team": ["GWS", "Collingwood"], "away_team": ["Geelong", "Carlton"],
                          "home_score": [100, 81], "away_score": [90, 70]})
    assert cross_check(stats, g) == {"matches": 2, "not_in_squiggle": 0, "score_mismatches": 1}
    monkeypatch.delenv("CEB_CONTACT_EMAIL", raising=False)
    with pytest.raises(RuntimeError, match="CEB_CONTACT_EMAIL"):
        user_agent()
    monkeypatch.setenv("CEB_CONTACT_EMAIL", "me@example.org")
    assert "me@example.org" in user_agent()
