"""A hash-chained copy of the sealed-table access log, in UTC: supplement/access_log_chain.jsonl.

    python scripts/access_log_chain.py            # write the chain and its head
    python scripts/access_log_chain.py --check    # recompute the chain from the supplement copy and compare
    ots stamp supplement/access_log_chain_head.txt  # timestamp the head (only its hash leaves the machine)

Every entry of prereg/access_log.jsonl, in order, with the account name replaced by "author" (as
scripts/public_access_log.py), the local US Pacific time converted to UTC, and two fields: "prev", the SHA-256 of the
previous entry's line ("0" * 64 for the first), and "sha256", that of this entry's canonical JSON with "prev" included.
Changing any earlier entry changes every later hash and the head; once the head is timestamped, a change made after
the timestamp is detectable. It cannot show that entries made before the timestamp were not edited before it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from invex.paths import REPO

SRC = REPO / "prereg" / "access_log.jsonl"
DST = REPO / "supplement" / "access_log_chain.jsonl"
HEAD = REPO / "supplement" / "access_log_chain_head.txt"


def canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain(rows: list[dict]) -> list[dict]:
    out, prev = [], "0" * 64
    for r in rows:
        e = {"table": r["table"], "reason": r["reason"], "user": "author",
             "time_utc": datetime.fromisoformat(r["time"]).replace(tzinfo=ZoneInfo("America/Los_Angeles"))
             .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "prev": prev}
        e["sha256"] = hashlib.sha256(canonical(e).encode()).hexdigest()
        out.append(e)
        prev = e["sha256"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the supplement copy's own hashes")
    a = ap.parse_args()
    if a.check:
        rows = [json.loads(line) for line in DST.read_text().splitlines() if line.strip()]
        prev = "0" * 64
        for i, e in enumerate(rows):
            body = {k: v for k, v in e.items() if k != "sha256"}
            if e["prev"] != prev or hashlib.sha256(canonical(body).encode()).hexdigest() != e["sha256"]:
                raise SystemExit(f"entry {i}: the chain is broken")
            prev = e["sha256"]
        head = HEAD.read_text().split()[0]
        if head != prev:
            raise SystemExit("the head does not match the last entry")
        print(f"{len(rows)} entries, chain intact, head {head}")
        return
    rows = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    c = chain(rows)
    DST.write_text("".join(canonical(e) + "\n" for e in c))
    HEAD.write_text(f"{c[-1]['sha256']}  head of supplement/access_log_chain.jsonl, {len(c)} entries, last "
                    f"{c[-1]['time_utc']}\n")
    print(f"{DST.relative_to(REPO)}: {len(c)} entries; head {c[-1]['sha256']}")


if __name__ == "__main__":
    main()
