"""Does the replay fill a virtual order like the market filled the real one? (appendix validation)

    python experiments/fill_validation.py --days 2019-01-30 2019-07-30 --stocks 200

For real orders that join the touch during continuous trading, a virtual twin with the same side, price
and size is inserted (a) at the real order's own queue position and (b) right behind it, and tracked with
the exact fill rule for up to 30 minutes. Under price-time priority the twin at the real order's position
must first fill exactly when the real order is first executed, and the twin behind it never earlier.
Writes results/fill_validation/{rows.parquet, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from invex import schema as S
from invex.eval.universe import load_splits
from invex.io.store import day_dir, read_group, symbols
from invex.paths import RESULTS
from invex.provenance import provenance, write_json
from invex.sim.probe import twin_probes
from invex.sim.tracker import prepare

NS = 1_000_000_000


def task(args):
    day, stock, n, seed = args
    qa = json.loads((day_dir("itch", day) / "qa.json").read_text())
    o, c = qa["session"]["open"], qa["session"]["close"]
    sym = symbols("itch", day).set_index("stock")
    book = prepare(read_group("itch", day, int(sym.loc[stock, "rg"])), o, c)
    i = np.arange(book.n)
    cont = (book.flags & S.F_CONTINUOUS) > 0
    prev_bid = np.r_[-1, book.bid_px[:-1]]
    prev_ask = np.r_[-1, book.ask_px[:-1]]
    joins_touch = ((book.side > 0) & (book.px == prev_bid)) | ((book.side < 0) & (book.px == prev_ask))
    cand = i[(book.kind == S.ADD) & cont & joins_touch & (book.ts < c - 40 * 60 * NS)]
    if cand.size == 0:
        return None
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(cand, size=min(n, cand.size), replace=False))
    t_max = book.ts[rows] + 30 * 60 * NS
    # the real order's first execution (EXEC rows that refer to it), within the same horizon
    ex = i[book.kind == S.EXEC]
    first_exec = pd.Series(ex, index=book.oid[ex]).groupby(level=0).min()
    real = first_exec.reindex(rows).to_numpy()
    real = np.where(np.isfinite(real), real, -1).astype(np.int64)
    real = np.where((real >= 0) & (book.ts[np.maximum(real, 0)] <= t_max), real, -1)
    twin0 = twin_probes(book, rows, 0, t_max)
    twin1 = twin_probes(book, rows, 1, t_max)
    return pd.DataFrame({"day": day, "stock": stock, "row": rows, "real_first": real, "twin_at": twin0,
                         "twin_behind": twin1, "qty": book.qty[rows]})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", nargs="+", default=["2019-01-30", "2019-07-30"])
    ap.add_argument("--stocks", type=int, default=200)
    ap.add_argument("--per-stock", type=int, default=300)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    sp = load_splits()
    train = sorted(s for s, v in sp["split"].items() if v == "train")
    stocks = sorted(np.random.default_rng(0).choice(train, size=min(a.stocks, len(train)), replace=False))
    jobs = [(d, s, a.per_stock, k) for k, (d, s) in enumerate((d, s) for d in a.days for s in stocks)
            if s in set(symbols("itch", d)["stock"])]
    env = os.environ.copy()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex_:
            df = pd.concat([r for r in ex_.map(task, jobs, chunksize=2) if r is not None], ignore_index=True)
    finally:
        os.environ.clear()
        os.environ.update(env)
    out = RESULTS / "fill_validation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "rows.parquet", index=False)
    executed = df["real_first"] >= 0
    summary = {
        "orders": len(df), "stock_days": int(df.groupby(["day", "stock"]).ngroups),
        "real_executed_share": float(executed.mean()),
        "twin_at_matches_real_first_execution": float((df["twin_at"][executed] == df["real_first"][executed]).mean()),
        "twin_at_fills_when_real_never_executed": float((df["twin_at"][~executed] >= 0).mean()),
        "twin_behind_fills_before_real": float(((df["twin_behind"] >= 0) & (df["twin_behind"] < df["real_first"]))[executed].mean()),
        "twin_behind_fills_when_real_executed": float((df["twin_behind"][executed] >= 0).mean()),
        "provenance": provenance(args=vars(a)),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
