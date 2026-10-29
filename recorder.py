"""
Coinbase tick recorder
======================

Records live top-of-book + trade size to CSV, in exactly the format
replay.py / calibrate.py expect:

    timestamp,bid,ask,bid_size,ask_size,volume

Uses the Coinbase EXCHANGE feed (wss://ws-feed.exchange.coinbase.com),
not Advanced Trade. Reason: the Exchange `ticker` channel is public — no
API key, no JWT — and one message carries best_bid, best_ask, both sizes,
and last_size. Advanced Trade's level2 requires auth and more plumbing for
the same information.

Run it and walk away. Two weeks of data is worth more than any amount of
parameter guessing.

    pip install websockets
    python recorder.py --product BTC-USD --out ./data

Ctrl-C to stop cleanly (flushes and closes the file).
"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import signal
import sys
from datetime import datetime, timezone

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

WS_URL = "wss://ws-feed.exchange.coinbase.com"
HEADER = ["timestamp", "bid", "ask", "bid_size", "ask_size", "volume"]


def parse_ticker(msg: dict) -> list | None:
    """
    Turn one Coinbase ticker message into a CSV row.
    Returns None for anything unusable — subscriptions, heartbeats,
    crossed books, missing fields.
    """
    if msg.get("type") != "ticker":
        return None

    try:
        bid = float(msg["best_bid"])
        ask = float(msg["best_ask"])
        bid_sz = float(msg.get("best_bid_size") or 0)
        ask_sz = float(msg.get("best_ask_size") or 0)
        vol = float(msg.get("last_size") or 0)
    except (KeyError, TypeError, ValueError):
        return None

    if bid <= 0 or ask <= bid:
        return None          # crossed or locked — never record it

    ts_raw = msg.get("time")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            ts = datetime.now(timezone.utc).timestamp()
    else:
        ts = datetime.now(timezone.utc).timestamp()

    return [f"{ts:.6f}", bid, ask, bid_sz, ask_sz, vol]


class Recorder:
    def __init__(self, product: str, out_dir: str, flush_every: int = 200):
        self.product = product
        self.out_dir = out_dir
        self.flush_every = flush_every
        os.makedirs(out_dir, exist_ok=True)

        self._fh = None
        self._writer = None
        self._day = None
        self.rows = 0
        self.dropped = 0
        self.running = True

    def _path_for_today(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.out_dir, f"{self.product}_{day}.csv"), day

    def _ensure_file(self):
        """Roll to a new file at UTC midnight so days stay separable."""
        path, day = self._path_for_today()
        if self._day == day and self._fh is not None:
            return
        if self._fh:
            self._fh.close()
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", buffering=1)
        self._writer = csv.writer(self._fh)
        if is_new:
            self._writer.writerow(HEADER)
        self._day = day
        print(f"[recorder] writing → {path}")

    def write(self, row: list):
        self._ensure_file()
        self._writer.writerow(row)
        self.rows += 1
        if self.rows % self.flush_every == 0:
            self._fh.flush()
            print(f"[recorder] {self.rows:,} rows  ({self.dropped} dropped)")

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    async def run(self):
        """
        Reconnecting loop. Disconnects are routine on a multi-day record —
        the point is that a dropped socket costs you seconds, not the session.
        """
        backoff = 1
        sub = json.dumps({
            "type": "subscribe",
            "product_ids": [self.product],
            "channels": ["ticker", "heartbeat"],
        })

        while self.running:
            try:
                async with websockets.connect(
                    WS_URL, ping_interval=20, ping_timeout=20, max_size=2 ** 22
                ) as ws:
                    await ws.send(sub)
                    print(f"[recorder] connected — {self.product}")
                    backoff = 1

                    async for raw in ws:
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            self.dropped += 1
                            continue

                        if msg.get("type") == "error":
                            print(f"[recorder] feed error: {msg.get('message')}")
                            continue

                        row = parse_ticker(msg)
                        if row:
                            self.write(row)
                        elif msg.get("type") == "ticker":
                            self.dropped += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.running:
                    break
                print(f"[recorder] disconnected ({type(e).__name__}) — retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

        self.close()
        print(f"[recorder] stopped. {self.rows:,} rows written.")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="BTC-USD",
                    help="BTC-USD, ETH-USD, SOL-USD …")
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()

    rec = Recorder(args.product, args.out)

    loop = asyncio.get_running_loop()

    def stop():
        print("\n[recorder] shutting down…")
        rec.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:
            pass          # Windows

    await rec.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
