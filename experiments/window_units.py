"""Units from a 15- or 60-minute window instead of the registered 30 (review round 4, R6; exploratory, validation).

    python experiments/window_units.py build --window 900 --source train_v0 --out train_w900   # training transitions
    python experiments/window_units.py build-val --window 900                                   # validation episodes
    python experiments/transfer.py fit --data train_w900 --tag w900 --reps R3tick --seeds 0 1 2
    python experiments/window_units.py eval --window 900 --models 'zero_R3tick_s[0-2]' 'w900_R3tick_s[0-2]'

The tasks stay exactly as frozen (start time, side, size Q and horizon T, which the registered 30-minute window set):
only the trailing-window statistics from which a policy's units are computed are recomputed over the other length,
from the same replayed book (invex.book.calib, as invex.env.episodes). At evaluation the policy sees units from the
other window while every cost is still normalised by the registered window's spread, as for the shuffled-units placebo
of transfer.py, so costs are comparable with the registered evaluation. 'build' writes a transitions set whose
episodes carry the other window's statistics (the transitions themselves, in physical units, are linked unchanged),
so a policy can also be trained with those units.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import OUT, TRANS, eval_task, pool  # noqa: E402

from invex.book.calib import calibrate  # noqa: E402
from invex.env.units import WINDOW_COLUMNS, unit_frame, window_columns  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval.freeze import book_of  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402


def rewindow_task(args):
    venue, day, stock, t0s, window = args
    book, o, c, tick = book_of(venue, day, stock)
    v = venue_of(venue)
    cal = calibrate(book, o, c, tick, v.px_scale, v.qty_scale)
    rows = [window_columns(cal.window(int(t), window)) for t in t0s]
    return venue, day, stock, pd.DataFrame(rows)


def rewindow(eps: pd.DataFrame, window: int, workers: int) -> pd.DataFrame:
    eps = eps.reset_index(drop=True).copy()
    groups = list(eps.groupby(["venue", "day", "stock"]).groups.items())
    jobs = [(v, d, s, eps.loc[idx, "t0"].to_numpy(), window) for (v, d, s), idx in groups]
    parts = pool(rewindow_task, jobs, workers)
    cols = [c for c in WINDOW_COLUMNS if c != "tick"]
    for ((v, d, s), idx), (_v, _d, _s, w) in zip(groups, parts):
        eps.loc[idx, cols] = w[cols].to_numpy()
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--window", type=int, required=True)
    b.add_argument("--source", default="train_v0")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=20)
    bv = sub.add_parser("build-val")
    bv.add_argument("--window", type=int, required=True)
    bv.add_argument("--workers", type=int, default=20)
    e = sub.add_parser("eval")
    e.add_argument("--window", type=int, required=True)
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "build":
        src, dst = TRANS / a.source, TRANS / a.out
        if dst.exists():
            raise SystemExit(f"{dst} exists")
        eps = rewindow(pd.read_parquet(src / "eps.parquet"), a.window, a.workers)
        dst.mkdir(parents=True)
        eps.to_parquet(dst / "eps.parquet", index=False)
        os.symlink((src / "data.npz").resolve(), dst / "data.npz")
        meta = json.loads((src / "meta.json").read_text())
        meta.update({"name": a.out, "window_s": a.window, "source": a.source,
                     "note": "the source's transitions with the trailing-window statistics recomputed over window_s",
                     "provenance": provenance(args=vars(a))})
        write_json(dst / "meta.json", meta)
        print(f"{a.out}: {len(eps):,} episodes, window {a.window}s, {time.time() - t0:.0f}s")
    elif a.cmd == "build-val":
        eps = rewindow(freeze.load("val_v0"), a.window, a.workers)
        eps.to_parquet(OUT / f"window_val_w{a.window}.parquet", index=False)
        print(f"val_v0 with {a.window}s windows: {len(eps):,} episodes, {time.time() - t0:.0f}s")
    else:
        ko = load_splits()["ko_constants"]
        true_eps = freeze.load("val_v0")
        eps = pd.read_parquet(OUT / f"window_val_w{a.window}.parquet")
        assert (eps["key"].to_numpy() == true_eps["key"].to_numpy()).all()
        paths = sorted({str(p) for pat in a.models for p in (OUT / "models").glob(f"{pat}.pkl")})
        jobs = [(d, s, g, paths, ko, "exact", None) for (_v, d, s), g in eps.groupby(["venue", "day", "stock"])]
        res = pd.concat(pool(eval_task, jobs, a.workers), ignore_index=True)
        keep = ["key", "venue", "day", "stock", "Q", "side", "family", "cell", "w_spread", "w_mid", "tick"]
        res = res.merge(true_eps[keep], on="key")  # costs normalised by the registered window's units
        res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
        res["c_bps"] = 1e4 * res["cost"] / (res["Q"] * res["w_mid"])
        out = f"val_v0_w{a.window}"
        res.to_parquet(OUT / f"eval_{out}.parquet", index=False)
        write_json(OUT / f"eval_{out}.json", {"episodes": "val_v0", "window_s": a.window,
                                              "models": [os.path.basename(p) for p in paths], "rows": len(res),
                                              "seconds": round(time.time() - t0), "provenance": provenance(args=vars(a))})
        print(res.groupby("policy")["c_spread"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
