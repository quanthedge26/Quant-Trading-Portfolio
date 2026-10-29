"""
Kalshi orderbook recorder
=========================

Records top-of-book for one or more Kalshi markets to CSV, in the same
shape the replay/calibrate harness reads:

    timestamp,ticker,bid,ask,bid_size,ask_size,spread_cents,mid

THE IMPORTANT BIT — Kalshi's orderbook is not like a normal exchange.
It returns two separate sides, both expressed as BIDS:

    orderbook.yes  ->  people bidding to BUY YES
    orderbook.no   ->  people bidding to BUY NO

There is no "ask" array. You derive it, because buying NO at 58c is
mathematically identical to selling YES at 42c:

    best YES bid  =  highest price in orderbook.yes
    best YES ask  =  100 - (highest price in orderbook.no)

Get that conversion wrong and every number downstream is garbage. This
is the single most common Kalshi API mistake.

Prices are in CENTS (1-99). A contract settles at 0 or 100.

    pip install requests cryptography
    export KALSHI_KEY_ID="your-key-id"
    export KALSHI_PRIVATE_KEY="/path/to/kalshi-private-key.pem"
    python kalshi_recorder.py --tickers KXFED-26DEC-T3.25 --out ./kalshi_data

Ctrl-C stops cleanly.

NOTE: Kalshi's API surface and auth requirements change. If you get a 401
or 404, check the current docs at docs.kalshi.com and adjust BASE / the
endpoint path below. The parsing logic is the part that matters and it is
independent of the URL.
"""

from __future__ import annotations
import argparse
import base64
import csv
import os
import signal
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError:
    sys.exit("pip install cryptography")


BASE = "https://api.elections.kalshi.com/trade-api/v2"
HEADER = ["timestamp", "ticker", "bid", "ask", "bid_size", "ask_size",
          "spread_cents", "mid"]


# ---------------------------------------------------------------------
# Auth — Kalshi signs (timestamp + METHOD + path) with RSA-PSS
# ---------------------------------------------------------------------

def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def sign(private_key, timestamp_ms: str, method: str, path: str) -> str:
    msg = (timestamp_ms + method.upper() + path).encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def auth_headers(key_id: str, private_key, method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sign(private_key, ts, method, path),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------
# Orderbook parsing — the part that actually matters
# ---------------------------------------------------------------------

def top_of_book(orderbook: dict) -> dict | None:
    """
    Convert Kalshi's two-sided bid structure into a normal bid/ask.

    orderbook = {"yes": [[price, size], ...], "no": [[price, size], ...]}
    Prices are cents. Either side can be null or empty on a thin market.
    """
    if not isinstance(orderbook, dict):
        return None

    yes = orderbook.get("yes") or []
    no = orderbook.get("no") or []

    def best(levels):
        # Highest price is the best bid on that side
        rows = [(int(p), int(s)) for p, s in levels
                if p is not None and s is not None]
        return max(rows, key=lambda r: r[0]) if rows else (None, None)

    yes_bid, yes_bid_sz = best(yes)
    no_bid, no_bid_sz = best(no)

    # A NO bid at p is a YES ask at 100 - p
    yes_ask = (100 - no_bid) if no_bid is not None else None
    yes_ask_sz = no_bid_sz

    if yes_bid is None or yes_ask is None:
        return None                      # one-sided book, skip the tick
    if yes_ask <= yes_bid:
        return None                      # crossed/locked, skip

    return {
        "bid": yes_bid,
        "ask": yes_ask,
        "bid_size": yes_bid_sz,
        "ask_size": yes_ask_sz,
        "spread_cents": yes_ask - yes_bid,
        "mid": (yes_bid + yes_ask) / 2.0,
    }


# ---------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------

class KalshiRecorder:
    def __init__(self, tickers, out_dir, interval, key_id=None, key_path=None):
        self.tickers = tickers
        self.out_dir = out_dir
        self.interval = interval
        os.makedirs(out_dir, exist_ok=True)

        self.key_id = key_id
        self.private_key = load_private_key(key_path) if key_path else None
        self.authed = bool(key_id and key_path)

        self.session = requests.Session()
        self._fh = None
        self._writer = None
        self._day = None
        self.rows = 0
        self.skipped = 0
        self.errors = 0
        self.running = True

    # --- file handling ---
    def _ensure_file(self):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._day == day and self._fh:
            return
        if self._fh:
            self._fh.close()
        path = os.path.join(self.out_dir, f"kalshi_{day}.csv")
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", buffering=1)
        self._writer = csv.writer(self._fh)
        if new:
            self._writer.writerow(HEADER)
        self._day = day
        print(f"[kalshi] writing -> {path}")

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    # --- api ---
    def fetch_orderbook(self, ticker: str) -> dict | None:
        path = f"/trade-api/v2/markets/{ticker}/orderbook"
        url = BASE + f"/markets/{ticker}/orderbook"
        headers = {}
        if self.authed:
            headers = auth_headers(self.key_id, self.private_key, "GET", path)
        try:
            r = self.session.get(url, headers=headers, timeout=10)
            if r.status_code == 401:
                print("[kalshi] 401 — check KALSHI_KEY_ID / private key")
                return None
            if r.status_code == 404:
                print(f"[kalshi] 404 — ticker not found: {ticker}")
                return None
            if r.status_code == 429:
                print("[kalshi] rate limited, backing off")
                time.sleep(2)
                return None
            r.raise_for_status()
            return r.json().get("orderbook")
        except Exception as e:
            self.errors += 1
            if self.errors % 10 == 1:
                print(f"[kalshi] request error ({type(e).__name__})")
            return None

    # --- main loop ---
    def run(self):
        print(f"[kalshi] polling {len(self.tickers)} market(s) "
              f"every {self.interval}s  (auth={'yes' if self.authed else 'no'})")
        while self.running:
            cycle_start = time.time()
            for ticker in self.tickers:
                if not self.running:
                    break
                ob = self.fetch_orderbook(ticker)
                if ob is None:
                    continue
                top = top_of_book(ob)
                if top is None:
                    self.skipped += 1
                    continue
                self._ensure_file()
                self._writer.writerow([
                    f"{time.time():.3f}", ticker,
                    top["bid"], top["ask"],
                    top["bid_size"], top["ask_size"],
                    top["spread_cents"], f'{top["mid"]:.1f}',
                ])
                self.rows += 1

            if self.rows and self.rows % 100 < len(self.tickers):
                print(f"[kalshi] {self.rows:,} rows  "
                      f"({self.skipped} skipped, {self.errors} errors)")

            elapsed = time.time() - cycle_start
            time.sleep(max(0.0, self.interval - elapsed))

        self.close()
        print(f"[kalshi] stopped. {self.rows:,} rows written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True,
                    help="comma-separated market tickers")
    ap.add_argument("--out", default="./kalshi_data")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between polling cycles (2s is plenty — "
                         "prediction markets do not tick like crypto)")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    key_id = os.environ.get("KALSHI_KEY_ID")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY")
    if not (key_id and key_path):
        print("[kalshi] no credentials found — trying unauthenticated.")
        print("         set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY if you get 401s.")

    rec = KalshiRecorder(tickers, args.out, args.interval, key_id, key_path)

    def stop(*_):
        print("\n[kalshi] shutting down…")
        rec.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    rec.run()


if __name__ == "__main__":
    main()
