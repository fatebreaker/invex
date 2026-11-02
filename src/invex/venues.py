"""Venues: the scales of their event tables, order latency, and the task families that make sense there.

Event tables keep each venue's own integer units (ITCH and IEX: 1e-4 $ and shares; Coinbase: one tick of the
pair-day and 1e-8 coins, so its price scale is per symbol-day).
Everything dimensionless is unaffected. Two things are not: Kyle-Obizhaeva's W, which needs dollars and
shares (or coins), and the raw-unit representations (R0: 1 s, 100 shares, $0.01; R0desc's asset
descriptors), which are defined in the ITCH frame and converted per row (`to_ref`). The physical task family
P (100 or 300 shares) is equity-only: a share and a coin are not the same amount of anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    name: str
    px_scale: float | None  # price units per dollar (None: per symbol-day, from the symbols table)
    qty_scale: float | None  # size units per share (or coin)
    latency_ns: int  # decision to exchange
    families: tuple[str, ...]


ITCH = Venue("itch", 10_000, 1, 100_000, ("MS", "KO", "P"))
COINBASE = Venue("coinbase", None, None, 5_000_000, ("MS", "KO"))  # prices in ticks of each pair-day
# IEX: the ITCH price and size units; 100 us as on Nasdaq plus the 350 us speed bump on every incoming order
IEX = Venue("iex", 10_000, 1, 450_000, ("MS", "KO", "P"))
VENUES = {v.name: v for v in (ITCH, COINBASE, IEX)}
REF = ITCH  # the frame of raw units and descriptors


def venue(name: str) -> Venue:
    """The venue's constants; synthetic venues (e.g. 'zi') use the ITCH frame."""
    return VENUES.get(name, ITCH)


def to_ref(px_scale, qty_scale):
    """(price factor, size factor) that convert venue units to the reference frame (1 for ITCH)."""
    return REF.px_scale / px_scale, REF.qty_scale / qty_scale
