"""The public, code-only copy of the repository: one commit with the code at HEAD and no history.

    python scripts/export_public_code.py [--parent <commit>]    # prints the new commit's id; pushes nothing

The commit holds src/, experiments/, scripts/, tests/, pyproject.toml and .gitignore as they are at HEAD, with
release/README.md as its README. The registrations, the paper, the supplement and the talks are left out. Every file
goes through scripts/export_anonymous.py's scrub (absolute paths become placeholders, the environment script's venv
becomes $INVEX_VENV), and the export fails if any of that script's identifying strings remains (the account and home
paths, git author names and e-mails, session links, the untracked .anon_patterns). The author is anonymous and the
time is in UTC. With --parent, the commit continues that commit (the public branch's current head), so that a plain
push fast-forwards it. The commit is built in a temporary index; the working tree and the branches are untouched.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_anonymous import identifying, scrub  # noqa: E402

from invex.paths import REPO  # noqa: E402

KEEP = ("src/", "experiments/", "scripts/", "tests/")
TOP = ("pyproject.toml", ".gitignore")
RENAME = {"release/README.md": "README.md"}


def git(*args: str, env: dict | None = None, data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, env=env, input=data, capture_output=True, check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", default="", help="the public branch's current head")
    ap.add_argument("--message", default="invex: exact L3 replay engine, execution environments, learners and evaluation")
    a = ap.parse_args()
    pats, bad, n = identifying(), [], 0
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        for row in git("ls-tree", "-r", "HEAD").decode().splitlines():
            meta, path = row.split("\t", 1)
            mode, _, sha = meta.split()
            out = RENAME.get(path, path if path.startswith(KEEP) or path in TOP else None)
            if out is None:
                continue
            text = git("cat-file", "blob", sha).decode()
            new = scrub(path, text)
            for p in pats:
                for m in p.finditer(new):
                    bad.append(f"{path}:{new.count(chr(10), 0, m.start()) + 1}: {m.group(0)}")
            if new != text:
                sha = git("hash-object", "-w", "--stdin", data=new.encode()).decode().strip()
            git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{out}", env=env)
            n += 1
        if bad:
            raise SystemExit("identifying strings remain:\n  " + "\n  ".join(bad))
        tree = git("write-tree", env=env).decode().strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000")
    ident = {"GIT_AUTHOR_NAME": "Anonymous", "GIT_AUTHOR_EMAIL": "anonymous@example.com", "GIT_AUTHOR_DATE": now,
             "GIT_COMMITTER_NAME": "Anonymous", "GIT_COMMITTER_EMAIL": "anonymous@example.com", "GIT_COMMITTER_DATE": now}
    parent = ["-p", a.parent] if a.parent else []
    msg = a.message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    commit = git("commit-tree", tree, *parent, env={**os.environ, **ident}, data=msg.encode()).decode().strip()
    print(f"{n} files, no identifying strings", file=sys.stderr)
    print(commit)


if __name__ == "__main__":
    main()
