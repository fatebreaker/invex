"""Where raw downloads, processed data and run outputs live (set in scripts/env.sh).

Raw files go to $INVEX_RAW (default /tmp/invex_raw: large but wiped at boot, so
every download is resumable). Processed data go to $INVEX_DATA (default
<repo>/data) on the shared /data volume, under a hard budget
($INVEX_DATA_BUDGET_GB, default 250) that every build checks before writing.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(os.environ.get("INVEX_REPO", Path(__file__).resolve().parents[2]))
RAW = Path(os.environ.get("INVEX_RAW", "/tmp/invex_raw"))
DATA = Path(os.environ.get("INVEX_DATA", REPO / "data"))
RESULTS = Path(os.environ.get("INVEX_RESULTS", REPO / "results"))
BUDGET_GB = float(os.environ.get("INVEX_DATA_BUDGET_GB", "250"))

GB = 1024**3


def dir_size_gb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except FileNotFoundError:  # removed while walking
                pass
    return total / GB


def ensure_space(path: Path, need_gb: float, budget_gb: float | None = None, budget_root: Path | None = None) -> None:
    """Raise before writing `need_gb` under `path` if the volume or the budget would be exceeded.

    Keeps at least 50 GB free on the volume (it is shared) and, when a budget
    is given, keeps the size of `budget_root` (default: `path`) under it.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / GB
    if free_gb - need_gb < 50:
        raise RuntimeError(f"only {free_gb:.0f} GB free on the volume of {path}; need {need_gb:.0f} GB + 50 GB headroom")
    if budget_gb is not None:
        used = dir_size_gb(budget_root or path)
        if used + need_gb > budget_gb:
            raise RuntimeError(f"{budget_root or path} holds {used:.0f} GB; adding {need_gb:.0f} GB exceeds the {budget_gb:.0f} GB budget")
