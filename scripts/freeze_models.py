"""Record the sha256 of every policy a pre-registration freezes: prereg/models_v1.json, prereg/models_iex_v1.json.

    python scripts/freeze_models.py [--check]
    python scripts/freeze_models.py --manifest prereg/models_iex_v1.json [--check]

prereg-v1 tags: zero, seent, cbseen (seeds 0-2), x<stat> and xref<stat> (seed 0). iex-v1: every zero, seent and
iexseen policy of seeds 0-2, and the neural zeronn policies of seeds 0-4, on disk when it is written. With --check,
verifies that the files on disk still match the recorded hashes (run before every sealed evaluation). iex-v2:
the PPO policies (zeroppo, seeds 0-2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from invex.paths import REPO, RESULTS

PATTERNS = ("zero_*_s[012].pkl", "seent_*_s[012].pkl", "cbseen_*_s[012].pkl", "xactivity_*_s0.pkl",
            "xrefactivity_*_s0.pkl", "xtradesize_*_s0.pkl", "xreftradesize_*_s0.pkl", "xspread_*_s0.pkl",
            "xrefspread_*_s0.pkl")
OUT = REPO / "prereg" / "models_v1.json"
MANIFESTS = {"models_v1.json": PATTERNS,
             "models_iex_v1.json": ("zero_*_s[012].pkl", "seent_*_s[012].pkl", "iexseen_*_s[012].pkl",
                                    "zeronn_*_s[01234].pkl"),
             "models_iex_v2.json": ("zeroppo_*_s[012].pkl",)}


def digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--manifest", default=str(OUT))
    a = ap.parse_args()
    out = REPO / "prereg" / a.manifest.split("/")[-1]
    models = RESULTS / "transfer" / "models"
    files = sorted({p for pat in MANIFESTS[out.name] for p in models.glob(pat)})
    if a.check:
        rec = json.loads(out.read_text())["models"]
        bad = [n for n, h in rec.items() if not (models / n).exists() or digest(models / n) != h]
        print(f"{len(rec)} recorded models; {len(bad)} missing or changed" + (f": {bad[:5]}" if bad else ""))
        sys.exit(1 if bad else 0)
    if out.exists():
        raise SystemExit(f"{out} exists: manifests are never rewritten")
    rec = {p.name: digest(p) for p in files}
    out.write_text(json.dumps({"models": rec, "count": len(rec)}, indent=1) + "\n")
    by_tag = {}
    for n in rec:
        by_tag[n.split("_")[0]] = by_tag.get(n.split("_")[0], 0) + 1
    print(f"{out}: {len(rec)} models {by_tag}")


if __name__ == "__main__":
    main()
