"""Two post hoc checks of the IEX test days for review round 2 (exploratory; one logged load of the sealed table).

    python experiments/iex_review2.py              # the IEX test days (post hoc; loads the sealed table, logged)
    python experiments/iex_review2.py --dry-run    # the same on the unsealed IEX validation days, written nowhere

1. The decisive NHP episode (22 October, a 300-share sell): the IEX book at each decision time and at the forced
   cross that the waiting policies met (t0 + T plus the venue latency): best bid and ask and their sizes, the mid
   against the arrival mid, the visible bid levels the cross walks (price and size, up to the replay's 50 non-empty
   levels), the price of the last one and the shares beyond it that the shortfall rule prices; and IEX's own trades
   during the episode. Everything comes from the replay's own functions (invex.sim.tracker).
2. P4 and the fine-tuned reference with seeds resampled: zero-shot PPO, PPO trained on IEX and the fine-tuned policy
   on the one-tick books (w_onetick >= 0.5), under both IEX estimands, with the registered two-way bootstrap over
   stocks and days and, within each arm, its three seeds resampled as well (95% intervals), and every seed pair.

Writes results/transfer/iex_review2.json and paper/numbers_iex_review2.tex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from invex import schema as S  # noqa: E402
from invex import venues  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.sim import tracker as tk  # noqa: E402

KEY = "ca7384a089980ae2be44"  # NHP, 22 October: the decisive episode of the stock-weighted fitted-Q tests
K = 10
ARMS = {"zero": "zeroppo_R3tick", "ix": "ixppo_R3tick", "ft": "ftppo_R3tick"}
PAIRS = {"PFour": ("zero", "ix"), "FtVsZero": ("ft", "zero"), "FtVsIx": ("ft", "ix")}


def bid_levels(b, j: int, q: int, max_levels: int = 50) -> list[tuple[int, int]]:
    """(price, size) of the non-empty visible bid levels after row j, best first, until q shares or max_levels."""
    n_lev = b.levels.shape[0]
    L = tk.level_index(b.levels, b.bid_px[j])
    out, rem = [], q
    while rem > 0 and L >= 0 and len(out) < max_levels:
        s = int(tk.size_at(b.lvl, b.lv_off, b.lv_rows, n_lev, 0, L, j))
        if s > 0:
            out.append((int(b.levels[L]), s))
            rem -= s
        L -= 1
    return out


def trace(eps: pd.DataFrame, key: str) -> dict:
    e = eps.loc[eps["key"] == key].iloc[0]
    book, _o, _c, tick = freeze.book_of("iex", str(e["day"]), str(e["stock"]))
    arrs, lat = book.arrays(), venues.IEX.latency_ns
    t0, T, Q, sgn, mid0 = int(e["t0"]), int(e["T_ns"]), int(e["Q"]), int(e["side"]), float(e["mid0"])
    usd = lambda p: p / venues.IEX.px_scale  # noqa: E731

    def state(j: int) -> dict:
        f, cash, used, short = tk.walk(*arrs, sgn, Q, j, 50, 5, tick)
        if f == 0 and short == Q:  # the replay prices an empty side off the near touch (execution._cross)
            cash = Q * ((book.bid_px[j] if sgn > 0 else book.ask_px[j]) + sgn * 6 * tick)
        return {"bid": usd(book.bid_px[j]), "bid_size": int(book.bid_sz[j]), "ask": usd(book.ask_px[j]),
                "ask_size": int(book.ask_sz[j]), "mid_move": usd((book.bid_px[j] + book.ask_px[j]) / 2 - mid0),
                "cross_filled": int(f), "cross_levels": int(used), "cross_short": int(short),
                "cross_cost_per_share": usd(sgn * (cash - mid0 * Q) / Q)}

    decisions = []
    for k in range(K):
        t_k = t0 + (T * k) // K
        decisions.append({"k": k, "t_s": (t_k - t0) / 1e9, **state(tk.row_after(book.ts, t_k + lat) - 1)})
    j_end = tk.row_after(book.ts, t0 + T + lat) - 1
    walked = bid_levels(book, j_end, Q)
    sel = (book.ts > t0) & (book.ts <= t0 + T + lat) & np.isin(book.kind, [S.EXEC, S.HIDDEN])
    at_aux = (book.kind[sel] == S.HIDDEN) | ((book.flags[sel] & S.F_EXEC_PRICE) > 0)  # the trade price is in aux
    trades = pd.DataFrame({"t_s": (book.ts[sel] - t0) / 1e9, "price": usd(np.where(at_aux, book.aux[sel], book.px[sel])),
                           "shares": book.qty[sel], "hidden": book.kind[sel] == S.HIDDEN})
    return {"stock": str(e["stock"]), "day": str(e["day"]), "t0_ns": t0, "T_s": T / 1e9, "Q": Q, "side": sgn,
            "mid0": usd(mid0), "tick": usd(tick),
            "w_onetick": float(e["w_onetick"]), "family": str(e["family"]), "decisions": decisions,
            "end": {**state(j_end), "levels": [(usd(p), s) for p, s in walked]},
            "trades": trades.to_dict("records"), "n_trades": int(len(trades))}


def seed_tables(res: pd.DataFrame, by: str):
    """Per arm, per seed: stock x day cell values (sums, or means weighted so each stock's days sum to one)."""
    res = res.assign(arm=res["policy"].str.replace(r"_s\d+$", "", regex=True),
                     seed=res["policy"].str.extract(r"_s(\d+)$")[0].astype(int))
    per = res.pivot_table(index=["key", "stock", "day"], columns=["arm", "seed"], values="c_spread")
    per = per.dropna()  # the episodes every arm and seed has
    cells = {}
    for (arm, seed), v in per.items():
        g = v.groupby(level=["stock", "day"])
        cells.setdefault(arm, {})[seed] = g.sum().unstack() if by == "episode" else g.mean().unstack()
    n = per.iloc[:, 0].groupby(level=["stock", "day"]).count().unstack()
    ref = next(iter(next(iter(cells.values())).values()))
    n = n.reindex_like(ref)
    if by == "episode":
        w = n.fillna(0.0).to_numpy()
    else:
        w = n.notna().astype(float)
        w = w.div(w.sum(axis=1), axis=0).to_numpy()
    arr = {arm: np.stack([d[s].reindex_like(ref).fillna(0.0).to_numpy() for s in sorted(d)]) for arm, d in cells.items()}
    return arr, w, int(len(per))


def boot(arr: dict, w: np.ndarray, x: str, y: str, by: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Estimate of x - y, its 95% interval with stocks, days and each arm's seeds resampled, and per seed pair."""
    X, Y = arr[ARMS[x]], arr[ARMS[y]]
    num = (lambda Z: Z) if by == "episode" else (lambda Z: Z * w)  # cell sums, or weighted cell means

    def est(zx, zy, wa, wd):
        return float(wa @ num(zx - zy) @ wd / max(wa @ w @ wd, 1e-12))

    A, D = w.shape
    one_a, one_d = np.ones(A), np.ones(D)
    point = est(X.mean(0), Y.mean(0), one_a, one_d)
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    for i in range(n_boot):
        wa = np.bincount(rng.integers(0, A, A), minlength=A).astype(float)
        wd = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        sx, sy = rng.integers(0, X.shape[0], X.shape[0]), rng.integers(0, Y.shape[0], Y.shape[0])
        b[i] = est(X[sx].mean(0), Y[sy].mean(0), wa, wd)
    pairs = {f"{i}{j}": est(X[i], Y[j], one_a, one_d) for i in range(X.shape[0]) for j in range(Y.shape[0])}
    lo, hi = np.quantile(b, [0.025, 0.975])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "pairs": pairs}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cols = ["key", "policy", "stock", "day", "c_spread", "excluded"]
    if a.dry_run:
        eps = freeze.load("iex_val_v0")
        evals = ("iex_val_v8", "iex_val_v0_ft")
        v8 = pd.read_parquet(RESULTS / "transfer" / "eval_iex_val_v8.parquet", columns=["key", "policy", "shortfall"])
        key = v8.loc[(v8["policy"] == "post_then_cross") & (v8["shortfall"] > 0), "key"].iloc[0]
    else:
        eps = freeze.load("iex_test_v0", unseal=True,
                          reason="post hoc (review round 2): the IEX book at the cross of the decisive NHP episode, "
                                 "and P4 and the fine-tuned reference with seeds resampled (exploratory)")
        evals, key = ("iex_test_v8", "iex_test_ft"), KEY
    out = {"nhp": trace(eps, key)}
    keep = set(eps.loc[eps["w_onetick"] >= 0.5, "key"])
    res = pd.concat([pd.read_parquet(RESULTS / "transfer" / f"eval_{e}.parquet", columns=cols)
                     for e in evals], ignore_index=True)
    res = res[~res["excluded"] & res["key"].isin(keep) & res["policy"].str.contains("R3tick")]
    res = res.drop_duplicates(["key", "policy"])  # zero-shot PPO is in both evaluations, identically
    m = {}
    for by, tag in (("stock", "Stock"), ("episode", "Episode")):
        arr, w, n_eps = seed_tables(res, by)
        out[f"episodes_{by}"] = n_eps
        for label, (x, y) in PAIRS.items():
            r = boot(arr, w, x, y, by)
            out[f"{label}_{by}"] = r
            m[f"IexRevSeed{label}{tag}"] = f"{r['mean']:+.3f}"
            m[f"IexRevSeed{label}{tag}Lo"], m[f"IexRevSeed{label}{tag}Hi"] = f"{r['lo']:+.3f}", f"{r['hi']:+.3f}"
            vals = list(r["pairs"].values())
            m[f"IexRevSeed{label}{tag}PairMin"], m[f"IexRevSeed{label}{tag}PairMax"] = f"{min(vals):+.3f}", f"{max(vals):+.3f}"
            print(f"{label:9s} {by:8s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] seed pairs "
                  f"{min(vals):+.4f} .. {max(vals):+.4f}")
    e = out["nhp"]["end"]
    m.update({"NhpMidZero": f"{out['nhp']['mid0']:.3f}", "NhpEndBid": f"{e['bid']:.2f}", "NhpEndAsk": f"{e['ask']:.2f}",
              "NhpEndMidMove": f"{e['mid_move']:+.3f}", "NhpEndLevels": str(len(e["levels"])),
              "NhpEndVisible": str(sum(s for _p, s in e["levels"])), "NhpEndShort": str(e["cross_short"]),
              "NhpEndLastLevel": f"{e['levels'][-1][0]:.2f}" if e["levels"] else "--",
              "NhpEndCostPerShare": f"{e['cross_cost_per_share']:.2f}", "NhpHorizon": f"{out['nhp']['T_s']:.0f}",
              "NhpTrades": str(out["nhp"]["n_trades"])})
    print(json.dumps({k: v for k, v in out["nhp"].items() if k not in ("trades",)}, indent=1, default=str)[:6000])
    if a.dry_run:
        return
    write_json(RESULTS / "transfer" / "iex_review2.json", {**out, "provenance": provenance()})
    lines = ["% generated by experiments/iex_review2.py (post hoc, exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / "numbers_iex_review2.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
