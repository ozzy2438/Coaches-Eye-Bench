"""The test-season lock: protocol tag, protocol integrity, frozen choices, single-run receipt."""

from __future__ import annotations

from pathlib import Path

from .config import PROTOCOL_PATH, ROOT
from .io_utils import git

TAG = "protocol-frozen"


class GuardError(RuntimeError):
    pass


def check_protocol_frozen(repo: Path = ROOT, protocol: Path = PROTOCOL_PATH) -> None:
    rc, _ = git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}", cwd=repo)
    if rc != 0:
        raise GuardError(f"git tag '{TAG}' does not exist: write, commit and tag protocol.md first")
    rel = str(Path(protocol).resolve().relative_to(Path(repo).resolve()))
    rc, out = git("diff", "--quiet", TAG, "--", rel, cwd=repo)
    if rc != 0:
        raise GuardError(f"{rel} differs from the '{TAG}' tag: record departures in DEVIATIONS.md, "
                         "or re-freeze (FREEZE_LOG.md) before any test data is loaded")


def check_test_allowed(results_dir: Path, repo: Path = ROOT, protocol: Path = PROTOCOL_PATH) -> None:
    check_protocol_frozen(repo, protocol)
    frozen = Path(results_dir) / "val" / "frozen_choices.json"
    if not frozen.exists():
        raise GuardError(f"{frozen} missing: run `make train` (selection on validation) first")
    receipt = Path(results_dir) / "test" / "test_run.json"
    if receipt.exists():
        raise GuardError(
            f"{receipt} exists: the test season has already been evaluated once. Further analysis "
            "of 2026 must go under results/post_hoc/ and be labelled post-hoc")
