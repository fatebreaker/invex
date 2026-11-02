"""Domain randomisation over time as well as price and size (review round 4, R5; exploratory, validation only).

    python experiments/time_dr.py gen --name tdr_v0            # time-stretched copies of the training stock-days
    python experiments/time_dr.py fit --reps R0r R0 --seeds 0 1 2 --device cuda
    python experiments/transfer.py eval --episodes val_v0 --models 'tdrnn_*' --out val_v0_tdr

The rescaling augmentation of invex.agents.fqi_nn (augment_scales) multiplies prices and sizes but cannot rescale
time, because the raw features count flow over windows fixed in seconds. Here time is rescaled where it can be done
consistently, in the replay itself: every event time of a training stock-day is stretched about the session open by a
factor c (log-uniform in [1/4, 4], one per stock-day), the session with it, and the agent's latency by the same factor.
The stretched day is a synthetic stock that trades c times slower in every respect (its trailing-window statistics,
its fill times and its task horizons are all computed from the stretched replay), so features, tasks and costs are
consistent. Episodes are drawn and branched exactly as transfer.py gen does. The policy is the neural learner on raw
features, trained on the training transitions plus their stretched copies, with prices and sizes also rescaled
(augment_scales), so it is randomised in time, size and price together; R0r keeps the cost in spreads (a common
reward scale), R0 in cents (as the earlier augmentation arm). Models: tdrnn_<rep>_s<seed>.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import K, OUT, TRANS, load_transitions, pool  # noqa: E402

from invex.agents import fqi_adv, fqi_nn  # noqa: E402
from invex.env.episodes import book_episodes, seed_of  # noqa: E402
from invex.env.features import FlowBook, r1_stats  # noqa: E402
from invex.env.units import unit_frame, valid_units  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.io.store import day_dir, read_group, symbols  # noqa: E402
from invex.paths import ensure_space  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.sim.tracker import EXACT, prepare  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402

SPAN = 4.0


def stretch_of(day: str, stock: str) -> float:
    rng = np.random.default_rng(seed_of("itch", day, stock, "time-dr-stretch"))
    return float(np.exp(rng.uniform(-np.log(SPAN), np.log(SPAN))))


def gen_task(args):
    day, stock, n, ko, tag = args
    venue = "itch"
    sym = symbols(venue, day).set_index("stock")
    qa = json.loads((day_dir(venue, day) / "qa.json").read_text())
    o, c, tick = qa["session"]["open"], qa["session"]["close"], 100
    ev = read_group(venue, day, int(sym.loc[stock, "rg"])).copy()
    s = stretch_of(day, stock)
    ev["ts"] = o + np.round((ev["ts"].astype(np.float64) - o) * s).astype(np.int64)
    c2 = o + int(round((c - o) * s))
    book = prepare(ev, o, c2)
    eps = book_episodes(book, venue, day, stock, o, c2, n, ko, seed_tag=tag, tick=tick)
    if eps.empty:
        return None
    uf = unit_frame(eps, ko)
    eps = eps[valid_units(uf)].reset_index(drop=True)
    if eps.empty:
        return None
    for col, v in r1_stats(FlowBook.build(book), eps, tick=tick).items():
        eps[col] = v
    eps["stretch"] = s
    lat = int(round(venue_of(venue).latency_ns * s))
    return eps, fqi_adv.generate_book(book, eps, K, lat, EXACT, tick, seed_of(day, stock, tag, "behaviour"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--name", default="tdr_v0")
    g.add_argument("--per-book", type=int, default=100)
    g.add_argument("--workers", type=int, default=16)
    f = sub.add_parser("fit")
    f.add_argument("--reps", nargs="+", default=["R0r", "R0"])
    f.add_argument("--seeds", nargs="+", type=int, default=[0])
    f.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ko = load_splits()["ko_constants"]
    t0 = time.time()
    if a.cmd == "gen":
        out = TRANS / a.name
        if (out / "data.npz").exists():
            raise SystemExit(f"{out} exists")
        src = pd.read_parquet(TRANS / "train_v0" / "eps.parquet", columns=["day", "stock"]).drop_duplicates()
        jobs = [(d, s, a.per_book, ko, a.name) for d, s in src.itertuples(index=False)]
        ensure_space(out, len(jobs) * a.per_book * 7e3 / 1024**3)
        parts = pool(gen_task, jobs, a.workers)
        eps = pd.concat([p[0] for p in parts], ignore_index=True)
        data = fqi_adv.concat([p[1] for p in parts])
        out.mkdir(parents=True, exist_ok=True)
        eps.to_parquet(out / "eps.parquet", index=False)
        np.savez(out / "data.npz", **data)
        write_json(out / "meta.json", {"name": a.name, "venue": "itch", "split": "train", "stretch_span": SPAN,
                                       "per_book": a.per_book, "episodes": len(eps), "books": len(parts),
                                       "seconds": round(time.time() - t0), "provenance": provenance(args=vars(a))})
        print(f"{a.name}: {len(eps):,} episodes from {len(parts)} stretched books in {time.time() - t0:.0f}s")
    else:
        eps, data = load_transitions(["train_v0", "tdr_v0"])
        for seed in a.seeds:
            dd, ee = fqi_nn.augment_scales(data, eps, seed)  # prices and sizes too
            uf = unit_frame(ee, ko)
            c_r0 = float(np.median(uf["R3"]["t"]))
            for rep in a.reps:
                path = OUT / "models" / f"tdrnn_{rep}_s{seed}.pkl"
                if path.exists():
                    continue
                m = fqi_nn.fit(dd, ee, uf, rep, K, c_r0, seed=seed, device=a.device, log=lambda s: None)
                m.save(path)
                print(f"tdrnn {rep} seed {seed}: {len(ee):,} episodes, {time.time() - t0:.0f}s -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
