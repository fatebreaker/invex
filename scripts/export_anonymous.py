"""Anonymised source export for double-blind review: the tracked files at HEAD, without git history (whose commit
metadata names the author), with the access log replaced by its anonymised copy and absolute paths in provenance
records replaced by placeholders. Fails if any identifying string remains: the account and home paths, git author
names and e-mails, session links, and any string listed in the untracked file .anon_patterns.

    python scripts/export_anonymous.py [--out DIR]

Writes DIR/invex-anon.tar.gz and DIR/invex-anon.zip, the same files (default DIR: the system temporary directory;
OpenReview takes a supplement as a ZIP), with the frozen registered policies listed in POLICIES under
supplement/policies/ unless --no-policies. Nothing is uploaded. The content hashes
of the frozen splits and episode tables cover the data, not these provenance fields, so they still verify; the
registration tags themselves can only be checked against the full repository after review.
"""

from __future__ import annotations

import argparse
import getpass
import io
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

from invex.paths import REPO

PREFIX = "invex/"
# frozen registered policies shipped with the review copy (untracked: they live in results/transfer/models), each
# checked against the SHA-256 its registration recorded (prereg/models_iex_v*.json): the registered PPO policies of the
# IEX test, the IEX-trained PPO reference of P4, and the tree policies of the main contrast (MS+tick) and of the
# IEX-trained fitted-Q reference. The other registered policies are rebuilt exactly by the code (deterministic fits).
POLICIES = ([f"zeroppo_{r}_s{s}.pkl" for r in ("R3tick", "R3", "R0", "R1") for s in range(3)]
            + [f"ixppo_R3tick_s{s}.pkl" for s in range(3)])
TREE_POLICIES = [f"zero_R3tick_s{s}.pkl" for s in range(3)] + [f"iexseen_R3tick_s{s}.pkl" for s in range(3)]
# untracked outputs added as they are (experiments/export_episode_outputs.py: per-episode costs of the main arms)
EXTRA_DIRS = ("supplement/episode_outputs",)
REGISTRATIONS = ("prereg/models_iex_v1.json", "prereg/models_iex_v2.json", "prereg/models_iex_v8.json")
EXCLUDE = ("talk/",)  # talks and the one-page summary carry the presenter's name; they are not part of the review copy


def identifying() -> list[re.Pattern]:
    user = getpass.getuser()
    names = {user, str(REPO), str(REPO.resolve()), str(Path.home())}
    names |= {n.strip() for n in subprocess.run(["git", "log", "--format=%an%n%ae"], cwd=REPO, capture_output=True,
                                                 text=True, check=True).stdout.splitlines() if n.strip()}
    # further strings to refuse (an affiliation, say), one per line, in the untracked file .anon_patterns
    extra = REPO / ".anon_patterns"
    if extra.exists():
        names |= {x.strip() for x in extra.read_text().splitlines() if x.strip() and not x.startswith("#")}
    pats = [re.escape(n) for n in names if len(n) >= 4]
    pats += [r"claude\.ai/code", r"session_[0]", r"/ho[m]e/", r"/data/ho[m]e"]  # bracketed: no self-match
    return [re.compile(p, re.IGNORECASE) for p in pats]


def scrub(path: str, text: str) -> str:
    """Placeholders for absolute paths in provenance records and the environment script."""
    for root in sorted({str(REPO), str(REPO.resolve())}, key=len, reverse=True):
        text = text.replace(root, "<repo>")
    text = text.replace(str(Path.home()), "<home>")
    if path == "scripts/env.sh":
        text = re.sub(r'export PATH="[^"]*venvs/invex/bin:\$PATH"', 'export PATH="${INVEX_VENV:-$HOME/venvs/invex}/bin:$PATH"',
                      text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()))
    ap.add_argument("--no-policies", action="store_true", help="leave the frozen policies out (a smaller archive)")
    ap.add_argument("--tree-policies", action="store_true",
                    help="also the registered MS+tick and IEX-reference tree policies (about 40 MB more)")
    a = ap.parse_args()
    staged = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    modes = {}
    for row in staged.decode().split("\0"):
        if row:
            meta, f = row.split("\t", 1)
            modes[f] = 0o755 if meta.split()[0] == "100755" else 0o644
    files = sorted(f for f in modes if not f.startswith(EXCLUDE))

    def show(f: str) -> bytes:
        return subprocess.run(["git", "show", f"HEAD:{f}"], cwd=REPO, capture_output=True, check=True).stdout

    n_pub = sum(1 for x in show("prereg/access_log_public.jsonl").decode().splitlines() if x.strip())
    n_log = sum(1 for x in show("prereg/access_log.jsonl").decode().splitlines() if x.strip())
    if n_pub != n_log:
        raise SystemExit(f"the public log has {n_pub} entries, the log {n_log}: run scripts/public_access_log.py, commit")
    pats, bad = identifying(), []
    a.out.mkdir(parents=True, exist_ok=True)
    dst, dzip = a.out / "invex-anon.tar.gz", a.out / "invex-anon.zip"
    with tarfile.open(dst, "w:gz") as tar, zipfile.ZipFile(dzip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            raw = show("prereg/access_log_public.jsonl" if f == "prereg/access_log.jsonl" else f)
            try:
                text = scrub(f, raw.decode())
                data = text.encode()
                for p in pats:
                    for m in p.finditer(text):
                        line = text.count("\n", 0, m.start()) + 1
                        bad.append(f"{f}:{line}: {m.group(0)}")
            except UnicodeDecodeError:
                data = raw
            info = tarfile.TarInfo(PREFIX + f)
            info.size, info.mode, info.mtime = len(data), modes[f], 0
            tar.addfile(info, io.BytesIO(data))
            zi = zipfile.ZipInfo(PREFIX + f, date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr, zi.compress_type = (0o100000 | modes[f]) << 16, zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)
        if not a.no_policies:
            import hashlib
            import json

            reg = {}
            for r in REGISTRATIONS:
                d = json.loads(show(r).decode())
                for name, v in d.get("models", d).items():
                    reg.setdefault(name, (v if isinstance(v, str) else v.get("sha256"), r))
            rows = []
            for name in POLICIES + (TREE_POLICIES if a.tree_policies else []):
                data = (REPO / "results" / "transfer" / "models" / name).read_bytes()
                h = hashlib.sha256(data).hexdigest()
                if reg.get(name, (None,))[0] != h:
                    raise SystemExit(f"{name}: its hash does not match the registration")
                for p in pats:
                    if p.search(data.decode("latin-1")):
                        bad.append(f"supplement/policies/{name}: {p.pattern}")
                f = f"supplement/policies/{name}"
                info = tarfile.TarInfo(PREFIX + f)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
                zi = zipfile.ZipInfo(PREFIX + f, date_time=(1980, 1, 1, 0, 0, 0))
                zi.external_attr, zi.compress_type = (0o100644) << 16, zipfile.ZIP_DEFLATED
                zf.writestr(zi, data)
                rows.append(f"| `{name}` | `{h}` | `{reg[name][1]}` |")
                files.append(f)
            readme = ("# Frozen registered policies\n\nEach file's SHA-256 equals the one its registration recorded (the "
                      "third column). Load them with `invex.agents.fqi_adv.load_model` and score them with "
                      "`experiments/transfer.py eval` on rebuilt data. The registered tree and neural policies "
                      "are rebuilt exactly by `experiments/transfer.py fit` (deterministic fits; their hashes are in the "
                      "same registration files).\n\n| file | sha256 | registered in |\n"
                      "|---|---|---|\n" + "\n".join(rows) + "\n").encode()
            f = "supplement/policies/README.md"
            info = tarfile.TarInfo(PREFIX + f)
            info.size, info.mode, info.mtime = len(readme), 0o644, 0
            tar.addfile(info, io.BytesIO(readme))
            zi = zipfile.ZipInfo(PREFIX + f, date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr, zi.compress_type = (0o100644) << 16, zipfile.ZIP_DEFLATED
            zf.writestr(zi, readme)
            files.append(f)
        for d in EXTRA_DIRS:
            for src in sorted((REPO / d).glob("*.parquet")):  # the README is tracked
                data = src.read_bytes()
                for p in pats:
                    if p.search(data.decode("latin-1")):
                        bad.append(f"{d}/{src.name}: {p.pattern}")
                f = f"{d}/{src.name}"
                info = tarfile.TarInfo(PREFIX + f)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
                zi = zipfile.ZipInfo(PREFIX + f, date_time=(1980, 1, 1, 0, 0, 0))
                zi.external_attr, zi.compress_type = (0o100644) << 16, zipfile.ZIP_STORED
                zf.writestr(zi, data)
                files.append(f)
    if bad:
        dst.unlink(), dzip.unlink()
        raise SystemExit("identifying strings remain:\n  " + "\n  ".join(bad[:50]))
    print(f"{dst}, {dzip.name}: {len(files)} files, no identifying strings")


if __name__ == "__main__":
    main()
