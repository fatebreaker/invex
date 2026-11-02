"""What produced a result: code version, command, libraries, host, inputs.

Every results or build JSON carries `provenance()` so a number in the paper can
be traced to the commit, the command and the data that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from .paths import REPO


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> dict:
    """Commit, whether the tree differs from it, a hash of the changes to tracked files (None if there are none)
    and the untracked, non-ignored files: 'dirty' with no diff hash means only untracked files differ."""
    diff = _git("diff", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all").splitlines()
    return {
        "commit": _git("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None,
        "untracked": sorted(line[3:] for line in status if line.startswith("??")),
    }


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "numba", "torch", "pyarrow", "pandas", "scipy", "lightgbm"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", "?")
    nb = sys.modules.get("numba")
    if nb is not None:
        out["numba_threads"] = nb.get_num_threads()
        try:
            out["numba_layer"] = nb.threading_layer()
        except ValueError:  # no parallel function has run yet
            out["numba_layer"] = os.environ.get("NUMBA_THREADING_LAYER", "")
    return out


def provenance(**extra) -> dict:
    return {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_state(),
        "versions": versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **extra,
    }


def file_sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        n = 0
        while buf := fh.read(1 << 24):
            h.update(buf)
            n += len(buf)
            if limit is not None and n >= limit:
                break
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (a crash mid-write never leaves a truncated file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the checkpoint
