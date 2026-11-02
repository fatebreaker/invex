"""Nasdaq TotalView-ITCH 5.0: download, parse, annotate and build per-symbol event tables."""

# The public sample days on https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/ (checked 2026-10-14).
# The 2018 files and 05302019 listed there are 1,245-byte 404 pages, so they are left out.
# The two itch50_05_* names carry no year; 2026 is inferred (check against the stock directory).
ITCH_DAYS: dict[str, str] = {
    "2019-01-30": "01302019.NASDAQ_ITCH50.gz",
    "2019-03-27": "03272019.NASDAQ_ITCH50.gz",
    "2019-07-30": "07302019.NASDAQ_ITCH50.gz",
    "2019-08-30": "08302019.NASDAQ_ITCH50.gz",
    "2019-10-18": "S101819-v50.txt.gz",
    "2019-10-30": "10302019.NASDAQ_ITCH50.gz",
    "2019-12-30": "12302019.NASDAQ_ITCH50.gz",
    "2020-01-30": "01302020.NASDAQ_ITCH50.gz",
    "2021-07-13": "S071321-v50.txt.gz",
    "2021-08-13": "S081321-v50.txt.gz",
    "2025-11-28": "S112825-v50.txt.gz",  # half day (market closes 13:00)
    "2025-12-08": "S120825-v50.txt.gz",
    "2025-12-09": "S120925-v50.txt.gz",
    "2025-12-10": "S121025-v50.txt.gz",
    "2025-12-11": "S121125-v50.txt.gz",
    "2025-12-12": "S121225-v50.txt.gz",
    "2026-05-15": "itch50_05_15.gz",
    "2026-05-18": "itch50_05_18.gz",
    "2026-06-12": "S061226-v50.txt.gz",
}

BASE_URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
