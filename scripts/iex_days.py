"""The test days of the IEX prospective test (prereg/iex_test_days_v1.json) and the checks the run script makes.

    python scripts/iex_days.py --check          # every listed day starts after the iex-v1 tag
    python scripts/iex_days.py --check --tag iex-v2   # ... after the iex-v2 tag (the PPO policies)
    python scripts/iex_days.py --check --qa     # ... and the days in use passed the book checks
    python scripts/iex_days.py --resolve        # print the days in use (registered rule below)
    python scripts/iex_days.py --spec prereg/iex_retro_days_v4.json --absent   # iex-v4 part A: none downloaded yet
    python scripts/iex_days.py --spec prereg/iex_retro_days_v4.json --resolve
    python scripts/iex_days.py --interim        # iex-v4 part B: the first five listed days that pass, if five do

Registered rule: the test days are the ten listed days; a listed day whose HIST file is missing or whose build fails
the book checks is replaced by the first reserve day (in order) that is available and passes. Every day, listed or
reserve, must open (09:30 New York) after the time of the iex-v1 tag.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from invex.io.store import day_dir
from invex.paths import REPO

SPEC = REPO / "prereg" / "iex_test_days_v1.json"
NY = ZoneInfo("America/New_York")


def tag_time(tag: str = "iex-v1") -> datetime:
    out = subprocess.run(["git", "-C", str(REPO), "for-each-ref", "--format=%(taggerdate:iso-strict)",
                          f"refs/tags/{tag}"], capture_output=True, text=True, check=True).stdout.strip()
    if not out:
        raise SystemExit(f"no {tag} tag")
    return datetime.fromisoformat(out)


def opens(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(hour=9, minute=30, tzinfo=NY)


def passed(day: str) -> bool:
    p = day_dir("iex", day) / "book_qa.json"
    return p.exists() and bool(json.loads(p.read_text())["passed"])


def resolve(spec: dict) -> list[str]:
    days, reserve = list(spec["days"]), list(spec["reserve"])
    out = []
    for d in days:
        if passed(d):
            out.append(d)
            continue
        while reserve:
            r = reserve.pop(0)
            if passed(r):
                out.append(r)
                break
    if len(out) != len(days):
        raise SystemExit(f"only {len(out)} usable days")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--tag", default="iex-v1")
    ap.add_argument("--spec", default=str(SPEC), help="a day list (iex-v4 part A: prereg/iex_retro_days_v4.json)")
    ap.add_argument("--absent", action="store_true", help="fail if any listed or reserve day is already on disk")
    ap.add_argument("--interim", action="store_true", help="print the first five listed days that pass, or fail")
    a = ap.parse_args()
    spec = json.loads(open(a.spec).read())
    if a.absent:
        from invex.paths import RAW
        there = [d for d in spec["days"] + spec["reserve"]
                 if day_dir("iex", d).exists() or (RAW / "iex" / d).exists()]
        if there:
            sys.exit(f"already on disk: {there}")
        print("none of the days is on disk")
    if a.interim:
        ok = [d for d in spec["days"] if passed(d)][:5]
        if len(ok) < 5:
            sys.exit(f"only {len(ok)} listed days pass so far")
        print(" ".join(ok))
    if a.check:
        t = tag_time(a.tag)
        early = [d for d in spec["days"] + spec["reserve"] if opens(d) <= t]
        if early:
            sys.exit(f"days that opened before the tag ({t.isoformat()}): {early}")
        if a.qa:
            print("days in use:", " ".join(resolve(spec)))
        print(f"tag {t.isoformat()}: every test day opens after it")
    if a.resolve:
        print(" ".join(resolve(spec)))


if __name__ == "__main__":
    main()
