"""Anonymised copy of the sealed-table access log for release: prereg/access_log_public.jsonl.

    python scripts/public_access_log.py

Keeps every entry, its table, time and reason, in order; replaces the account name with "author".
"""

from __future__ import annotations

import json

from invex.paths import REPO

src = REPO / "prereg" / "access_log.jsonl"
dst = REPO / "prereg" / "access_log_public.jsonl"
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
for r in rows:
    if "user" in r:
        r["user"] = "author"
dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"{dst.relative_to(REPO)}: {len(rows)} entries")
