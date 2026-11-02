"""Nasdaq TotalView-ITCH 5.0 message layout (specification v5.0).

In the binary files every message is preceded by a 2-byte big-endian length.
Every message starts with: type (1 byte), stock locate (2), tracking number
(2), timestamp (6, ns since midnight). Integers are big-endian; prices are
4-byte unsigned with 4 implied decimals (MWCB levels: 8 bytes, 8 decimals).
Offsets below count from the type byte.
"""

from __future__ import annotations

import struct

LENGTH = {
    "S": 12, "R": 39, "H": 25, "Y": 20, "L": 26, "V": 35, "W": 12, "K": 28,
    "J": 35, "h": 21, "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19,
    "U": 35, "P": 44, "Q": 40, "B": 19, "I": 50, "N": 20, "O": 48,
}
# as a byte-indexed table for compiled code: LEN_BY_CODE[ord(t)] (0 = unknown type)
LEN_BY_CODE = [0] * 256
for _t, _n in LENGTH.items():
    LEN_BY_CODE[ord(_t)] = _n

# common header
OFF_LOCATE, OFF_TRACKING, OFF_TS = 1, 3, 5

# field offsets
A_REF, A_SIDE, A_SHARES, A_STOCK, A_PRICE = 11, 19, 20, 24, 32  # also F (+ attribution at 36)
E_REF, E_SHARES, E_MATCH = 11, 19, 23
C_REF, C_SHARES, C_MATCH, C_PRINTABLE, C_PRICE = 11, 19, 23, 31, 32
X_REF, X_SHARES = 11, 19
D_REF = 11
U_OLD, U_NEW, U_SHARES, U_PRICE = 11, 19, 27, 31
P_REF, P_SIDE, P_SHARES, P_STOCK, P_PRICE, P_MATCH = 11, 19, 20, 24, 32, 36
Q_SHARES, Q_STOCK, Q_PRICE, Q_MATCH, Q_TYPE = 11, 19, 27, 31, 39
B_MATCH = 11
H_STOCK, H_STATE, H_REASON = 11, 19, 21
h_STOCK, h_MARKET, h_ACTION = 11, 19, 20
J_STOCK, J_REF, J_UPPER, J_LOWER, J_EXT = 11, 19, 23, 27, 31
S_EVENT = 11

# Stock Directory 'R'
R_FIELDS = (
    # name, offset, struct format
    ("stock", 11, "8s"),
    ("market_category", 19, "c"),
    ("financial_status", 20, "c"),
    ("round_lot", 21, ">I"),
    ("round_lots_only", 25, "c"),
    ("issue_class", 26, "c"),
    ("issue_subtype", 27, "2s"),
    ("authenticity", 29, "c"),
    ("short_sale_threshold", 30, "c"),
    ("ipo_flag", 31, "c"),
    ("luld_tier", 32, "c"),
    ("etp_flag", 33, "c"),
    ("etp_leverage", 34, ">I"),
    ("inverse", 38, "c"),
)

PRICE_SCALE = 10_000  # 4 implied decimals


# --- builders used by the tests (and handy for debugging) ------------------------------

def _hdr(t: str, locate: int, ts: int, tracking: int = 0) -> bytes:
    return t.encode() + struct.pack(">HH", locate, tracking) + ts.to_bytes(6, "big")


def _sym(stock: str) -> bytes:
    return stock.encode().ljust(8)


def frame(msg: bytes) -> bytes:
    """Prefix a message with its 2-byte length, as in the binary files."""
    return struct.pack(">H", len(msg)) + msg


def msg_S(ts: int, event: str) -> bytes:
    return _hdr("S", 0, ts) + event.encode()


def msg_R(locate: int, ts: int, stock: str, market_category: str = "Q", round_lot: int = 100,
          issue_class: str = "C", authenticity: str = "P", etp: str = "N") -> bytes:
    return (_hdr("R", locate, ts) + _sym(stock) + market_category.encode() + b"N"
            + struct.pack(">I", round_lot) + b"N" + issue_class.encode() + b"Z " + authenticity.encode()
            + b"N" + b"N" + b"1" + etp.encode() + struct.pack(">I", 0) + b"N")


def msg_H(locate: int, ts: int, stock: str, state: str, reason: str = "    ") -> bytes:
    return _hdr("H", locate, ts) + _sym(stock) + state.encode() + b" " + reason.encode()[:4].ljust(4)


def msg_h(locate: int, ts: int, stock: str, action: str, market: str = "Q") -> bytes:
    return _hdr("h", locate, ts) + _sym(stock) + market.encode() + action.encode()


def msg_A(locate: int, ts: int, ref: int, side: str, shares: int, stock: str, price: int, mpid: str | None = None) -> bytes:
    body = struct.pack(">Q", ref) + side.encode() + struct.pack(">I", shares) + _sym(stock) + struct.pack(">I", price)
    if mpid is None:
        return _hdr("A", locate, ts) + body
    return _hdr("F", locate, ts) + body + mpid.encode()[:4].ljust(4)


def msg_E(locate: int, ts: int, ref: int, shares: int, match: int) -> bytes:
    return _hdr("E", locate, ts) + struct.pack(">QIQ", ref, shares, match)


def msg_C(locate: int, ts: int, ref: int, shares: int, match: int, printable: bool, price: int) -> bytes:
    return _hdr("C", locate, ts) + struct.pack(">QIQ", ref, shares, match) + (b"Y" if printable else b"N") + struct.pack(">I", price)


def msg_X(locate: int, ts: int, ref: int, shares: int) -> bytes:
    return _hdr("X", locate, ts) + struct.pack(">QI", ref, shares)


def msg_D(locate: int, ts: int, ref: int) -> bytes:
    return _hdr("D", locate, ts) + struct.pack(">Q", ref)


def msg_U(locate: int, ts: int, old: int, new: int, shares: int, price: int) -> bytes:
    return _hdr("U", locate, ts) + struct.pack(">QQII", old, new, shares, price)


def msg_P(locate: int, ts: int, side: str, shares: int, stock: str, price: int, match: int, ref: int = 0) -> bytes:
    return _hdr("P", locate, ts) + struct.pack(">Q", ref) + side.encode() + struct.pack(">I", shares) + _sym(stock) + struct.pack(">IQ", price, match)


def msg_Q(locate: int, ts: int, shares: int, stock: str, price: int, match: int, cross_type: str) -> bytes:
    return _hdr("Q", locate, ts) + struct.pack(">Q", shares) + _sym(stock) + struct.pack(">IQ", price, match) + cross_type.encode()


def msg_B(locate: int, ts: int, match: int) -> bytes:
    return _hdr("B", locate, ts) + struct.pack(">Q", match)


def msg_J(locate: int, ts: int, stock: str, ref_price: int, upper: int, lower: int, ext: int = 0) -> bytes:
    return _hdr("J", locate, ts) + _sym(stock) + struct.pack(">IIII", ref_price, upper, lower, ext)


def msg_Y(locate: int, ts: int, stock: str, action: str = "0") -> bytes:
    return _hdr("Y", locate, ts) + _sym(stock) + action.encode()
