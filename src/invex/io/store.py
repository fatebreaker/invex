"""Event tables on disk: one Parquet file per (venue, day), one row group per symbol.

    $INVEX_DATA/<venue>/<YYYY-MM-DD>/events.parquet   the event tables (invex.schema)
    $INVEX_DATA/<venue>/<YYYY-MM-DD>/symbols.parquet  one row per symbol: attributes, `rg` (row group), counts
    $INVEX_DATA/<venue>/<YYYY-MM-DD>/qa.json          build checks and provenance

`load_events(venue, day, symbol)` reads one symbol's rows as a numpy record array.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import schema as S
from ..paths import DATA

ARROW_SCHEMA = pa.schema([(name, pa.from_numpy_dtype(S.EVENT_DTYPE[name])) for name in S.COLUMNS])

WRITE_OPTS = dict(
    compression="zstd",
    compression_level=3,
    use_dictionary=["kind", "side", "flags"],
    column_encoding={"ts": "DELTA_BINARY_PACKED", "oid": "DELTA_BINARY_PACKED", "px": "DELTA_BINARY_PACKED"},
    write_statistics=False,
)


def day_dir(venue: str, day: str) -> Path:
    return DATA / venue / day


class EventWriter:
    """Append one symbol's columns at a time; each symbol becomes one row group."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.writer = pq.ParquetWriter(self.tmp, ARROW_SCHEMA, **WRITE_OPTS)
        self.n_groups = 0

    def write(self, cols: dict[str, np.ndarray]) -> int:
        tbl = pa.Table.from_arrays([pa.array(cols[c]) for c in S.COLUMNS], schema=ARROW_SCHEMA)
        self.writer.write_table(tbl, row_group_size=max(1, tbl.num_rows))
        self.n_groups += 1
        return self.n_groups - 1

    def close(self) -> None:
        self.writer.close()
        self.tmp.replace(self.path)


@lru_cache(maxsize=64)
def _pqfile(path: str) -> pq.ParquetFile:
    return pq.ParquetFile(path, memory_map=True)


@lru_cache(maxsize=64)
def symbols(venue: str, day: str) -> pd.DataFrame:
    return pd.read_parquet(day_dir(venue, day) / "symbols.parquet")


def load_events(venue: str, day: str, symbol: str) -> np.ndarray:
    """One symbol's event table as a structured array with dtype invex.schema.EVENT_DTYPE."""
    sym = symbols(venue, day)
    hit = sym.loc[sym["stock"] == symbol]
    if hit.empty:
        raise KeyError(f"{symbol} not in {venue} {day}")
    return read_group(venue, day, int(hit["rg"].iloc[0]))


def read_group(venue: str, day: str, rg: int) -> np.ndarray:
    tbl = _pqfile(str(day_dir(venue, day) / "events.parquet")).read_row_group(rg)
    out = np.empty(tbl.num_rows, dtype=S.EVENT_DTYPE)
    for c in S.COLUMNS:
        out[c] = tbl.column(c).to_numpy()
    return out
