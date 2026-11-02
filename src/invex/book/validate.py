"""Replay every symbol of a built day and check the book (invex.book.derive).

    python -m invex.book.validate --venue itch --day 2019-01-30 [--workers 32]

Writes $INVEX_DATA/<venue>/<day>/book_qa.parquet (one row per symbol) and
book_qa.json (totals, rates, worst symbols, pass/fail against THRESHOLDS).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .. import schema as S
from ..io.store import day_dir, read_group, symbols
from ..provenance import provenance, write_json

# a day passes when (rates over all symbols)
THRESHOLDS = {
    "unresolved": 0,  # every reference resolves
    "del_mismatch": 0,  # deletes remove exactly what was left
    "fifo_rate_min": 0.99,  # executions hit the oldest order at their level
    "crossed_rate_max": 1e-4,  # share of continuous-trading rows with a locked or crossed own book
    "ts_back": 0,  # rows are in time order within every table
}


def _session(venue: str, day: str) -> tuple[int, int]:
    qa = json.loads((day_dir(venue, day) / "qa.json").read_text())
    return qa["session"]["open"], qa["session"]["close"]


def check_symbol(args) -> dict:
    venue, day, rg, stock, open_ns, close_ns, tick = args
    from .derive import DOLLAR, PENNY, derive  # import in the worker (numba compiled once per process)

    ev = read_group(venue, day, rg)
    # ITCH: the penny grid above $1; Coinbase tables are in ticks of each pair-day, every price on the grid
    d = derive(ev, open_ns, close_ns, tick=tick, tick_floor=0) if tick else derive(ev, open_ns, close_ns, PENNY, DOLLAR)
    cont = (ev["flags"] & S.F_CONTINUOUS) > 0
    ex = (ev["kind"] == S.EXEC) & ((ev["flags"] & S.F_PRINTABLE) > 0)
    hid = ev["kind"] == S.HIDDEN
    return {
        "stock": stock, "rg": rg, "rows": len(ev), "rows_cont": int(cont.sum()),
        "trades_cont": int(((ev["flags"] & S.F_TRADE_START) > 0)[cont & ex].sum()),
        "exec_volume": int(ev["qty"][ex].sum()), "hidden_volume": int(ev["qty"][hid].sum()),
        "levels": int(d.levels.shape[0]), **d.qa,
    }


def validate(venue: str, day: str, workers: int = 32) -> dict:
    t0 = time.time()
    sym = symbols(venue, day).sort_values("n_rows", ascending=False)
    if venue == "coinbase":  # each pair's own stream: snapshot to last message
        jobs = [(venue, day, int(r.rg), r.stock, int(r.t_snap), int(r.t_end), int(r.tick)) for r in sym.itertuples()]
    else:
        open_ns, close_ns = _session(venue, day)
        jobs = [(venue, day, int(r.rg), r.stock, open_ns, close_ns, 0) for r in sym.itertuples()]
    env = os.environ.copy()
    os.environ["NUMBA_NUM_THREADS"] = "1"  # one symbol per process; parallelism comes from the pool
    try:
        # spawn, not fork: forking a process whose OpenMP runtime is up can hang
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(check_symbol, jobs, chunksize=4))
    finally:
        os.environ.clear()
        os.environ.update(env)
    df = pd.DataFrame(res)
    df.to_parquet(day_dir(venue, day) / "book_qa.parquet", index=False)
    tot = df[["rows", "rows_cont", "fifo_ok", "fifo_bad", "crossed", "unresolved", "negative", "off_tick",
              "del_mismatch", "exec_away", "ts_back"]].sum().astype(int).to_dict()
    fifo_rate = tot["fifo_ok"] / max(1, tot["fifo_ok"] + tot["fifo_bad"])
    crossed_rate = tot["crossed"] / max(1, tot["rows_cont"])
    checks = {
        "unresolved": tot["unresolved"] <= THRESHOLDS["unresolved"],
        "del_mismatch": tot["del_mismatch"] <= THRESHOLDS["del_mismatch"],
        "fifo_rate": fifo_rate >= THRESHOLDS["fifo_rate_min"],
        "crossed_rate": crossed_rate <= THRESHOLDS["crossed_rate_max"],
        "ts_back": tot["ts_back"] <= THRESHOLDS["ts_back"],
    }
    df["fifo_rate"] = df["fifo_ok"] / (df["fifo_ok"] + df["fifo_bad"]).clip(lower=1)
    worst = {
        "fifo": df.loc[df["fifo_ok"] + df["fifo_bad"] >= 100].nsmallest(10, "fifo_rate")[["stock", "fifo_rate", "fifo_bad"]].to_dict("records"),
        "crossed": df.nlargest(10, "crossed")[["stock", "crossed", "rows_cont"]].to_dict("records"),
        "negative": df.nlargest(5, "negative")[["stock", "negative"]].to_dict("records"),
    }
    out = {"day": day, "venue": venue, "symbols": len(df), "totals": tot, "fifo_rate": fifo_rate,
           "crossed_rate": crossed_rate, "checks": checks, "passed": all(checks.values()),
           "thresholds": THRESHOLDS, "worst": worst, "seconds": round(time.time() - t0), "provenance": provenance()}
    write_json(day_dir(venue, day) / "book_qa.json", out)
    print(f"{venue} {day}: {len(df)} symbols, fifo {fifo_rate:.5f}, crossed rate {crossed_rate:.2e}, "
          f"unresolved {tot['unresolved']}, negative {tot['negative']}, del_mismatch {tot['del_mismatch']}, "
          f"off_tick {tot['off_tick']}, exec_away {tot['exec_away']}, ts_back {tot['ts_back']} "
          f"-> {'PASS' if out['passed'] else 'FAIL'} ({out['seconds']}s)", flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venue", default="itch")
    ap.add_argument("--day", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    for day in a.day:
        validate(a.venue, day, a.workers)


if __name__ == "__main__":
    main()
