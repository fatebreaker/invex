"""Per-episode costs of the main arms on the sealed tests, for the review supplement (review round 4, R6).

    python experiments/export_episode_outputs.py [--out supplement/episode_outputs]

From the stored evaluations (no sealed table is read): for every analysed episode of the two Nasdaq stages and of the
prospective IEX test's one-tick population, the stock, day, side, task family and cell, and each main arm's cost in
spreads per share, averaged over its three seeds; the stocks the seen policy saw are flagged. Arms: Nasdaq, zero-shot
MS+tick, MS, raw units and z-scores and the seen MS+tick policy; IEX, zero-shot and IEX-trained fitted Q-iteration and
PPO (MS+tick), zero-shot raw units with each learner, and crossing at once. Written as zstd-compressed parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from invex.paths import DATA, REPO, RESULTS

OUT = RESULTS / "transfer"
SETS = {
    "stage1": (["stage1"], ["zero_R3tick", "zero_R3", "zero_R0", "zero_R1", "seent_R3tick"]),
    "stage2": (["stage2"], ["zero_R3tick", "zero_R3", "zero_R0", "zero_R1", "seent_R3tick"]),
    "iex_test": (["iex_test", "iex_test_ppo", "iex_test_v8"],
                 ["zero_R3tick", "iexseen_R3tick", "zeroppo_R3tick", "ixppo_R3tick", "zero_R0", "zeroppo_R0"]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "supplement" / "episode_outputs")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seen = set(pd.read_parquet(DATA / "transitions" / "testseen_v0" / "eps.parquet", columns=["stock"])["stock"])
    for name, (evals, arms) in SETS.items():
        pols = [f"{x}_s{s}" for x in arms for s in range(3)] + ["immediate"]
        parts, info = [], None
        for ev in evals:
            t = ds.dataset(OUT / f"eval_{ev}.parquet").to_table(
                columns=["key", "policy", "stock", "day", "side", "family", "cell", "c_spread", "excluded"],
                filter=ds.field("policy").isin(pols)).to_pandas()
            t = t[~t["excluded"]]
            parts.append(t[["key", "policy", "c_spread"]])
            i = t.drop_duplicates("key")[["key", "stock", "day", "side", "family", "cell"]]
            info = i if info is None else pd.concat([info, i]).drop_duplicates("key")
        r = pd.concat(parts, ignore_index=True).drop_duplicates(["key", "policy"])
        r["arm"] = r["policy"].str.replace(r"_s\d+$", "", regex=True)
        wide = r.pivot_table(index="key", columns="arm", values="c_spread", aggfunc="mean").astype("float32")
        df = info.set_index("key").join(wide, how="inner").reset_index(drop=True)
        if name.startswith("stage"):
            df["seen_stock"] = df["stock"].isin(seen)
        for c in ("stock", "day", "family", "cell"):
            df[c] = df[c].astype("category")
        df["side"] = df["side"].astype("int8")
        df = df.sort_values(["day", "stock"]).reset_index(drop=True)
        path = a.out / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        print(f"{path.relative_to(REPO)}: {len(df):,} episodes, {df.shape[1]} columns, "
              f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
