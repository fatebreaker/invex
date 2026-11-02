"""The sha256 of the policies iex-v8 adds (prereg/models_iex_v8.json): the IEX-trained PPO MS+tick policies.

    python scripts/freeze_models_v8.py [--check]

Without --check, writes the manifest (never rewritten). With --check, verifies that the files on disk still match the
recorded hashes (run before the iex-v8 evaluation, as scripts/freeze_models.py is for the earlier tags).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from invex.paths import REPO, RESULTS

OUT = REPO / "prereg" / "models_iex_v8.json"
NAMES = [f"ixppo_R3tick_s{s}.pkl" for s in (0, 1, 2)]


def digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    models = RESULTS / "transfer" / "models"
    if a.check:
        rec = json.loads(OUT.read_text())["models"]
        bad = [n for n, h in rec.items() if not (models / n).exists() or digest(models / n) != h]
        print(f"{len(rec)} recorded models; {len(bad)} missing or changed" + (f": {bad}" if bad else ""))
        sys.exit(1 if bad else 0)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: manifests are never rewritten")
    rec = {n: digest(models / n) for n in NAMES}
    OUT.write_text(json.dumps({"models": rec, "count": len(rec)}, indent=1) + "\n")
    print(f"{OUT}: {len(rec)} models")


if __name__ == "__main__":
    main()
