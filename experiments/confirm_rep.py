"""The families of the second IEX prospective test (iex-rep-v1, prereg/iex_replication_v1.md).

    python experiments/confirm_rep.py --spec prereg/confirmatory_iex_rep_v1.json --family rep_fqi --macro-prefix IexRepFqi

Runs a family of the spec on the evaluation the family names ("eval"), with the code of the first prospective test:
experiments/confirm_robust.py for a family with an "estimator" (every episode weighted equally, or the median over
stocks), experiments/confirm.py for one without (stocks, and each stock's days, weighted equally). Two things differ,
and neither changes a number those scripts compute:
  * a policy's tag may contain digits after its first letter (irmnn1, vrexnn1, gdronn10), so that their seeds are
    averaged as every other policy's are; the tags of the first test's policies parse as before;
  * the reason written to the access log names this registration and says whether the family is registered or
    secondary, instead of the fixed strings of confirm.py and confirm_robust.py.
Writes what those scripts write: results/transfer/confirm_<eval>_<family>.{json,md}, paper/numbers_confirm_<eval>_<family>.tex.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm  # noqa: E402
import confirm_robust  # noqa: E402

from invex.eval import freeze  # noqa: E402

TAG = "iex-rep-v1"
NAME = re.compile(r"^(?P<tag>[a-z][a-z0-9]*)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--eval", default="", help="override the family's evaluation (the dry run on validation)")
    ap.add_argument("--macro-prefix", default="")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args()
    fam = json.loads(open(a.spec).read())["families"][a.family]
    ev = a.eval or fam["eval"]
    role = "secondary" if fam.get("secondary") else "registered"
    load = freeze.load

    def logged_load(name, unseal=False, reason=""):
        return load(name, unseal=unseal, reason=f"{role} analysis of {ev}, family {a.family} ({TAG})" if reason else "")

    freeze.load = logged_load
    confirm.NAME = NAME
    argv = ["--spec", a.spec, "--family", a.family, "--eval", ev, "--macro-prefix", a.macro_prefix,
            "--n-boot", str(a.n_boot)]
    (confirm_robust.main if "estimator" in fam else confirm.main)(argv)


if __name__ == "__main__":
    main()
