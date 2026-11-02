"""The sha256 of every policy the second IEX prospective test evaluates (iex-rep-v1): prereg/models_iex_rep_v1.json.

    python scripts/freeze_models_rep.py [--check]

Without --check, writes the manifest (never rewritten). With --check, verifies that the files on disk still match the
recorded hashes (run before each evaluation, as scripts/freeze_models.py is for the earlier tags). The patterns are
those scripts/run_iex_rep.sh evaluates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from invex.paths import REPO, RESULTS

OUT = REPO / "prereg" / "models_iex_rep_v1.json"
S3, S5 = "_s[012].pkl", "_s[01234].pkl"
PATTERNS = {"models": [f"zero_{r}{S3}" for r in ("R3tick", "R0", "R0info", "R0pop", "R1")]
            + [f"zeronn_{r}{S5}" for r in ("R3tick", "R0", "R0r", "R0info", "R0pop")]
            + [f"zeroppo_{r}{S3}" for r in ("R3tick", "R0", "R0info", "R0pop")]
            + [f"{t}_R3tick{S3}" for t in ("iexseen", "ixppo")]
            + [f"tdrnn_R0r{S3}"] + [f"{t}_{r}{S3}" for t in ("irmnn1", "vrexnn1", "gdronn10") for r in ("R0r", "R0info")],
            "models_ctx": [f"ctxnn_R0r{S3}"]}


def digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files() -> dict:
    out = {}
    for sub, pats in PATTERNS.items():
        for pat in pats:
            hits = sorted((RESULTS / "transfer" / sub).glob(pat))
            if not hits:
                raise SystemExit(f"no policy matches {sub}/{pat}")
            out |= {f"{sub}/{p.name}": p for p in hits}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        rec = json.loads(OUT.read_text())["models"]
        base = RESULTS / "transfer"
        bad = [n for n, h in rec.items() if not (base / n).exists() or digest(base / n) != h]
        print(f"{len(rec)} recorded models; {len(bad)} missing or changed" + (f": {bad}" if bad else ""))
        sys.exit(1 if bad else 0)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: manifests are never rewritten")
    rec = {n: digest(p) for n, p in files().items()}
    OUT.write_text(json.dumps({"models": rec, "count": len(rec)}, indent=1) + "\n")
    print(f"{OUT}: {len(rec)} models")


if __name__ == "__main__":
    main()
