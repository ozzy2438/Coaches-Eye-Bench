"""`ceb` command line: build / train / eval-val / eval-test / report / smoke / card / guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import analysis, guard, pipeline
from .config import ROOT, Paths, load_params
from .io_utils import read_json


def _need(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"missing {path}: {hint}")
    return path


def cmd_build(a, P, paths):
    out = pipeline.build(paths, P, test=a.test)
    o = out["join"]["overall"]
    print(f"join rate rows {o['match_rate_rows']:.4f}, mass {o['match_rate_mass']:.4f}; "
          f"{out['join']['excluded_matches']} matches excluded; QA ok={out['qa']['ok']}")
    for s, d in out["join"]["per_season"].items():
        print(f"  {s}: matches {d['included_matches']}/{d['stats_matches']}, vote rows {d['vote_rows']}, "
              f"rate {d['match_rate_rows']:.4f}")


def _dev(paths):
    pm = pd.read_parquet(_need(paths.processed / "player_match.parquet", "run `make build`"))
    return pm, read_json(paths.processed / "join_report.json")


def cmd_train(a, P, paths):
    pm, rep = _dev(paths)
    out = analysis.run_selection(pm, rep, P, paths.results / "val")
    print("frozen choices:", out["choices"], "| effective start:", out["split"]["effective_start"])
    for n in out["split"]["notes"]:
        print("  note:", n)


def cmd_eval_val(a, P, paths):
    pm, rep = _dev(paths)
    analysis.run_validation(pm, rep, P, paths.results / "val")
    print("validation outputs written to", paths.results / "val")


def cmd_eval_test(a, P, paths):
    guard.check_test_allowed(paths.results)
    dev, _ = _dev(paths)
    test = pd.read_parquet(_need(paths.processed / "player_match_test.parquet", "run `make build-test`"))
    rep = read_json(paths.processed / "join_report_test.json")
    out = analysis.run_test(pd.concat([dev, test], ignore_index=True), rep, P, paths.results,
                            paths.results / "test", derived_dir=paths.data / "derived",
                            data_manifest=paths.data / "manifest_test.json")
    m = out["metrics"]
    print("TEST (single run):", {k: round(v["ndcg5"]["point"], 3) for k, v in m["metrics"].items()})
    print("decisions:", m["decisions"])


def cmd_report(a, P, paths):
    from .report import write_report
    readme = None if a.no_readme else ROOT / "README.md"
    print("wrote", write_report(Path(a.results) if a.results else paths.results, readme))


def cmd_smoke(a, P, paths):
    from .smoke import run_smoke
    out = Path(a.out or paths.results / "smoke")
    r = run_smoke(out)
    from .report import write_report
    write_report(out, None)
    print(f"smoke OK in {r['seconds']:.0f}s -> {out}")


def cmd_card(a, P, paths):
    from .card import build_card
    scores = pd.read_parquet(_need(Path(a.scores), "run the pipeline first"))
    print("wrote", build_card(scores, a.season, a.round, Path(a.out), synthetic=a.synthetic))


def cmd_manifest(a, P, paths):
    from .config import all_seasons
    from .manifest import build_manifest
    m = build_manifest(paths.raw, paths.manifest, seasons=all_seasons(P))
    print(f"manifest: {len(m['files'])} files -> {paths.manifest}")


def cmd_seasons(a, P, paths):
    s = P.splits
    print(f"{s.test_season}" if a.which == "test" else f"{s.train_start}:{max(s.val_seasons)}")


def cmd_smoke_real(a, P, paths):
    """Plumbing check on two real, already-fetched dev seasons (never the test season)."""
    import copy
    from types import SimpleNamespace

    from .config import smoke_params
    pm, rep = _dev(paths)
    last = max(P.splits.val_seasons)
    S = smoke_params()
    d = copy.deepcopy(vars(S))
    d["splits"] = SimpleNamespace(train_start=last - 1, train_end=last - 1, val_seasons=[last],
                                  test_season=P.splits.test_season, latest_start_allowed=last - 1)
    d["trends"] = SimpleNamespace(**{**vars(S.trends), "last_season_included": last, "block_width": 1})
    S = SimpleNamespace(**d)
    sub = pm[pm["season"].isin([last - 1, last])]
    rep = {**rep, "per_season": {k: v for k, v in rep["per_season"].items() if int(k) in (last - 1, last)}}
    out = paths.results / "smoke_real"
    analysis.run_validation(sub, rep, S, out)
    print("real-data smoke OK ->", out, "(plumbing only; not a result)")


def cmd_guard(a, P, paths):
    guard.check_protocol_frozen()
    print(f"ok: tag {guard.TAG} exists and protocol.md matches it")


def cmd_squiggle(a, P, paths):
    from . import squiggle
    from .ingest import ingest_stats
    stats_raw, _ = pipeline.load_raw(paths.raw, list(range(P.splits.train_start, max(P.splits.val_seasons) + 1)))
    stats, _ = ingest_stats(stats_raw)
    games = pd.concat([squiggle.games_frame(squiggle.fetch_year(y, paths.raw))
                       for y in sorted(stats["season"].unique())])
    print(squiggle.cross_check(stats, games))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ceb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--test", action="store_true"); b.set_defaults(f=cmd_build)
    sub.add_parser("train").set_defaults(f=cmd_train)
    sub.add_parser("eval-val").set_defaults(f=cmd_eval_val)
    sub.add_parser("eval-test").set_defaults(f=cmd_eval_test)
    r = sub.add_parser("report"); r.add_argument("--results"); r.add_argument("--no-readme", action="store_true")
    r.set_defaults(f=cmd_report)
    s = sub.add_parser("smoke"); s.add_argument("--out"); s.set_defaults(f=cmd_smoke)
    c = sub.add_parser("card")
    for k in ("--scores", "--out"):
        c.add_argument(k, required=True)
    c.add_argument("--season", type=int, required=True); c.add_argument("--round", type=int, required=True)
    c.add_argument("--synthetic", action="store_true"); c.set_defaults(f=cmd_card)
    sub.add_parser("guard").set_defaults(f=cmd_guard)
    sub.add_parser("manifest").set_defaults(f=cmd_manifest)
    se = sub.add_parser("seasons"); se.add_argument("which", choices=["dev", "test"]); se.set_defaults(f=cmd_seasons)
    sub.add_parser("smoke-real").set_defaults(f=cmd_smoke_real)
    sub.add_parser("squiggle-check").set_defaults(f=cmd_squiggle)
    a = ap.parse_args(argv)
    P, paths = load_params(), Paths()
    try:
        a.f(a, P, paths.ensure() if a.cmd not in ("guard", "card", "seasons") else paths)
    except (guard.GuardError, analysis.StopAndReport, FileNotFoundError) as e:
        sys.exit(f"STOP: {e}")


if __name__ == "__main__":
    main()
