"""A manifest of registration tags for an OpenTimestamps proof: anyone with the repository can rebuild it and check
the proof, which dates it by the Bitcoin block that includes it. Only the manifest's hash is sent to the calendar
servers (ots stamp); the manifest itself holds hashes only.

    python scripts/timestamp_manifest.py --tags prereg-v1 crypto-rep-v1 iex-v1 iex-v2 iex-v3 > m.txt
    ots stamp m.txt                       # writes m.txt.ots (pending until a block includes it, a few hours)
    ots upgrade m.txt.ots                 # later: completes the proof
    ots verify m.txt.ots                  # needs a Bitcoin node, or check the block on any explorer: ots info

To verify a committed manifest, rebuild it with the tags it lists and compare: the files must be identical.

Per tag: the tag object, commit and tree ids and the tagger date; then the SHA-256 of every file under prereg/ and
of the scripts that run the registered tests, as they are at that tag's commit. With --all, of every file in the tag's
tree (the replay, evaluation and analysis code included), so that a copy without the git history can be checked.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess

from invex.paths import REPO

SCRIPTS = ("experiments/confirm.py", "experiments/analyze_iex.py", "scripts/iex_days.py", "scripts/run_iex_test.sh",
           "scripts/run_iex_ppo.sh", "scripts/run_iex_wide.sh", "scripts/run_crypto_prospective.sh")


def git(*args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout
    return out if binary else out.decode().strip()


def manifest(tags: list[str], everything: bool = False) -> str:
    lines = ["# invex registration manifest: tags, commits and file hashes (scripts/timestamp_manifest.py)"]
    for t in tags:
        obj = git("rev-parse", f"refs/tags/{t}")
        commit = git("rev-parse", f"{t}^{{commit}}")
        tree = git("rev-parse", f"{t}^{{tree}}")
        date = git("for-each-ref", "--format=%(taggerdate:iso-strict)", f"refs/tags/{t}")
        lines.append(f"tag {t} object {obj} commit {commit} tree {tree} date {date}")
        files = [f for f in git("ls-tree", "-r", "--name-only", commit).splitlines()
                 if everything or f.startswith("prereg/") or f in SCRIPTS]
        for f in sorted(files):
            h = hashlib.sha256(git("show", f"{commit}:{f}", binary=True)).hexdigest()
            lines.append(f"  {h}  {f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--all", action="store_true", help="hash every file in each tag's tree")
    a = ap.parse_args()
    print(manifest(a.tags, a.all), end="")


if __name__ == "__main__":
    main()
