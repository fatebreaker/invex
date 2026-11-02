"""The event table every data source is converted to (ITCH, Coinbase, synthetic markets).

One table per (venue, symbol, day), rows in exchange processing order:

    ts     int64   ns since local midnight (ITCH) or since 00:00 UTC (Coinbase)
    kind   uint8   one of the KIND codes below
    side   int8    +1 bid / -1 ask: side of the resting order the row is about
    px     int64   price level of that order, in the venue's integer price units
    qty    int64   shares affected (DELETE / MOD_DEL: the shares that were left)
    oid    int32   row of the ADD / SNAP / MOD_ADD that created the order, -1 if none
    aux    int64   execution price (EXEC from ITCH 'C', HIDDEN, CROSS) or a state code
    flags  uint8   FLAG bits below

The `oid` column is what makes replay cheap: an order was added before row
`i` exactly when `oid < i`, so the queue position of a virtual order can be
tracked from the table alone, without an order book (see invex.sim.tracker).
ADD, SNAP and MOD_ADD rows point to themselves (`oid == row`).
"""

from __future__ import annotations

import numpy as np

# kinds
ADD = 1  # order added to the book
SNAP = 2  # order already resting when the table starts (emitted in time priority)
EXEC = 3  # resting order executed (qty shares)
CANCEL = 4  # part of a resting order cancelled; keeps priority
DELETE = 5  # resting order removed
MOD_DEL = 6  # replace, first half: the old order leaves the book
MOD_ADD = 7  # replace, second half: the new order joins at the back of its level
HIDDEN = 8  # execution against non-displayed liquidity (ITCH 'P'); px = aux = trade price
CROSS = 9  # auction cross (ITCH 'Q'); px = aux = cross price, qty = cross volume
BREAK = 10  # a trade was broken (ITCH 'B'); aux = match number
STATE = 11  # trading-state change; aux = state code (see STATE_* below)
RESEED = 12  # the book was resynchronised from a fresh snapshot (Coinbase gaps)

KIND_NAMES = {
    ADD: "ADD", SNAP: "SNAP", EXEC: "EXEC", CANCEL: "CANCEL", DELETE: "DELETE",
    MOD_DEL: "MOD_DEL", MOD_ADD: "MOD_ADD", HIDDEN: "HIDDEN", CROSS: "CROSS",
    BREAK: "BREAK", STATE: "STATE", RESEED: "RESEED",
}

# flags
F_PRINTABLE = 1  # execution counts toward volume (ITCH 'E', printable 'C', 'P')
F_EXEC_PRICE = 2  # executed at a price different from the order's (ITCH 'C')
F_TRADE_START = 4  # first execution of an aggressor run (set by invex.book.derive)
F_CONTINUOUS = 8  # inside continuous trading (set by invex.book.derive)
F_BROKEN = 16  # execution later broken
F_EXCLUDED = 32  # inside an exclusion window: halt, pause, cross, collar (derive)

# STATE codes in `aux`: ITCH trading-action states as their ASCII code
# ('H' halted, 'P' paused, 'Q' quotation only, 'T' trading), operational halts
# as 1000 + action ('H' halted / 'T' resumed), LULD auction collars as 2000,
# feed outages (Coinbase reconnects, until the book is reseeded and a 'T' row follows) as 3000.
STATE_OPHALT = 1000
STATE_COLLAR = 2000
STATE_OUTAGE = 3000

EVENT_DTYPE = np.dtype(
    [
        ("ts", np.int64),
        ("kind", np.uint8),
        ("side", np.int8),
        ("px", np.int64),
        ("qty", np.int64),
        ("oid", np.int32),
        ("aux", np.int64),
        ("flags", np.uint8),
    ]
)
COLUMNS = EVENT_DTYPE.names
