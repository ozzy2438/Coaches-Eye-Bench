from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default, sort_keys=False) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, "git not installed"
    return r.returncode, (r.stdout + r.stderr).strip()


def git_state(cwd: Path | None = None) -> dict:
    rc, sha = git("rev-parse", "HEAD", cwd=cwd)
    rc2, dirty = git("status", "--porcelain", cwd=cwd)
    return {"commit": sha if rc == 0 else None, "dirty": bool(dirty) if rc2 == 0 else None}
