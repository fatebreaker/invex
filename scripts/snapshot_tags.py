"""Copy, for each registration tag, every Python file at the tag that differs from the current commit into
prereg/snapshots/<tag>/, so that the code behind each registered analysis can be audited in the anonymous supplement.

    python scripts/snapshot_tags.py iex-v6 iex-v8

Each copied file can be checked against the tag's entry in the Bitcoin-stamped manifests of prereg/timestamps/.
src/invex/eval/stats.py is left out: at every tag it differs from the current file only in a docstring that named
another project (see prereg/snapshots/README.md). Files absent at HEAD are copied too.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from invex.paths import REPO

SKIP = {"src/invex/eval/stats.py"}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args()
    for tag in a.tags:
        files = [f for f in git("ls-tree", "-r", "--name-only", tag).decode().splitlines() if f.endswith(".py")]
        out, n = REPO / "prereg" / "snapshots" / tag, 0
        for f in files:
            if f in SKIP:
                continue
            old = git("show", f"{tag}:{f}")
            cur = subprocess.run(["git", "show", f"HEAD:{f}"], cwd=REPO, capture_output=True).stdout
            if old != cur:
                dst = out / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(old)
                n += 1
        print(f"{tag}: {n} of {len(files)} Python files differ from HEAD -> {out}")


if __name__ == "__main__":
    main()
