"""
Tick Replay Harness + Market Making Diagnostics
===============================================

Replays real historical tick data through the quoting engine and produces
the diagnostics a market making desk actually looks at.

The headline PnL number is close to useless on its own. What matters:

  1. MARKOUT PnL  — where did the mid go 1s / 5s / 30s AFTER each of your
     fills? This is the single most important market making metric. If your
     markouts are negative, you are being adversely selected: the people
     trading against you know something, and no amount of spread widening
     fixes it. You need a better signal or you need to not quote.

  2. PnL ATTRIBUTION — split total PnL into:
       spread capture  (the edge you theoretically earn per round trip)
       inventory PnL   (what your leftover position did to you)
       fees / rebates
     A strategy can show positive total PnL purely because it got lucky on
     inventory drift. That is not a business.

  3. QUOTE UPTIME — percentage of time you had a two-sided quote resting.
     Market maker rebate programs have uptime requirements. Low uptime also
     means your backtest fill count is fantasy.

DATA FORMATS SUPPORTED
----------------------
CSV with a header. Two shapes:

  Book snapshots (preferred):
    timestamp,bid,ask,bid_size,ask_size
    1710000000.123,68450.10,68450.60,1.4,0.9

  Trades only (book is synthesized — lower fidelity):
    timestamp,price,size,side
    1710000000.123,68450.60,0.05,buy

Coinbase: use the /products/{id}/candles REST endpoint for bars, or record
the level2 websocket to file for true book data. Kaiko/Tardis.dev sell
historical L2 if you don't want to record it yourself.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Iterator, Callable
import csv
import math
import statistics
import time

from market_maker import MarketMaker, MarketConfig, BookState, CRYPTO, FUTURES, EQUITY_OPTIONS
from venue import PaperAdapter, OrderManager, OMSConfig, Fill


# ============================================================
# DATA LOADING
# ============================================================

def load_book_csv(path: str, limit: Optional[int] = None) -> list[BookState]:
    """Load book-snapshot CSV. Rows must be time-ordered."""
    out: list[BookState] = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            try:
                bid, ask = float(row["bid"]), float(row["ask"])
            except (KeyError, ValueError):
                continue
            if ask <= bid or bid <= 0:
                continue          # drop crossed/garbage rows
            out.append(BookState(
                bid=bid, ask=ask,
                bid_size=float(row.get("bid_size") or 1.0),
                ask_size=float(row.get("ask_size") or 1.0),
                timestamp=float(row["timestamp"]),
            ))
    return out


def load_trades_csv(path: str, spread_bps: float = 5.0,
                    limit: Optional[int] = None) -> list[tuple[BookState, float]]:
    """
    Load trade prints and synthesize a book around each. Lower fidelity —
    you're inventing the spread — but usable when L2 isn't available.
    Returns (book, traded_volume) pairs.
    """
    out = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            try:
                px, sz = float(row["price"]), float(row["size"])
            except (KeyError, ValueError):
                continue
            half = px * spread_bps / 10_000 / 2
            side = (row.get("side") or "buy").lower()
            # Skew displayed size toward the passive side
            bs, asz = (0.5, 2.0) if side == "buy" else (2.0, 0.5)
            out.append((
                BookState(bid=px - half, ask=px + half, bid_size=bs,
                          ask_size=asz, timestamp=float(row["timestamp"])),
                sz,
            ))
    return out


def synthetic_ticks(n: int, start_px: float, vol_bps: float,
                    spread_bps: float, flow_autocorr: float = 0.6,
                    seed: int = 0) -> list[tuple[BookState, float]]:
    """
    Synthetic data WITH autocorrelated order flow.

    A pure random walk has no signal, so any market maker just earns spread
    and pays adverse selection — it can't distinguish a good model from a
    bad one. Real order flow is autocorrelated (large orders get worked in
    slices), which is exactly what the microprice signal is designed to pick
    up. Use this to sanity-check the engine, then move to real data.
    """
    import random
    rng = random.Random(seed)
    out = []
    px = start_px
    flow = 0.0

    for i in range(n):
        # Persistent order flow imbalance
        flow = flow_autocorr * flow + (1 - flow_autocorr) * rng.gauss(0, 1)
        drift = flow * vol_bps * 0.4 / 10_000
        shock = rng.gauss(0, vol_bps / 10_000)
        px *= math.exp(drift + shock)

        half = px * spread_bps / 10_000 / 2
        # Book sizes lean AGAINST the flow direction (that side is getting eaten)
        lean = max(0.15, 1.0 - abs(flow) * 0.4)
        bid_sz = rng.uniform(0.5, 3.0) * (lean if flow > 0 else 1.0)
        ask_sz = rng.uniform(0.5, 3.0) * (lean if flow < 0 else 1.0)
        # Volume must be on the same scale as book depth, otherwise the
        # queue never clears and you get a backtest with almost no fills.
        volume = abs(flow) * rng.uniform(0.4, 2.0)

        out.append((
            BookState(bid=round(px - half, 6), ask=round(px + half, 6),
                      bid_size=bid_sz, ask_size=ask_sz,
                      timestamp=1_700_000_000.0 + i * 0.25),
            volume,
        ))
    return out


# ============================================================
# DIAGNOSTICS
# ============================================================

@dataclass
class MarkedFill:
    fill: Fill
    mid_at_fill: float
    idx: int


@dataclass
class ReplayResult:
    symbol: str
    ticks: int
    fills: int
    total_pnl: float
    spread_pnl: float
    inventory_pnl: float
    fees: float
    final_position: float
    sharpe: float
    max_drawdown: float
    quote_uptime: float
    markouts: dict[str, float]
    equity_curve: list[float] = field(default_factory=list)
    oms_stats: dict = field(default_factory=dict)

    def report(self) -> str:
        L = []
        L.append("=" * 64)
        L.append(f"REPLAY — {self.symbol}   ({self.ticks:,} ticks)")
        L.append("=" * 64)
        L.append("")
        L.append("  PnL attribution")
        L.append(f"    spread capture   {self.spread_pnl:>12,.2f}")
        L.append(f"    inventory        {self.inventory_pnl:>12,.2f}")
        L.append(f"    fees / rebates   {self.fees:>12,.2f}")
        L.append(f"    {'-'*34}")
        L.append(f"    TOTAL            {self.total_pnl:>12,.2f}")
        L.append("")
        L.append("  Risk")
        L.append(f"    sharpe (ann.)    {self.sharpe:>12.2f}")
        L.append(f"    max drawdown     {self.max_drawdown:>12,.2f}")
        L.append(f"    final position   {self.final_position:>12,.4f}")
        L.append("")
        L.append("  Activity")
        L.append(f"    fills            {self.fills:>12,}")
        L.append(f"    quote uptime     {self.quote_uptime:>11.1%}")
        for k, v in self.oms_stats.items():
            L.append(f"    {k:<16} {v:>12}")
        L.append("")
        L.append("  Markout (mid move after fill, + = good)")
        if not self.markouts:
            L.append("    no fills to mark out")
        for h, v in self.markouts.items():
            flag = ""
            if v < -0.02:
                flag = "   <-- ADVERSELY SELECTED"
            L.append(f"    {h:<16} {v:>+12.4f} bps{flag}")
        L.append("=" * 64)
        return "\n".join(L)


def _markout(marked: list[MarkedFill], books: list[BookState],
             horizons_ticks: dict[str, int]) -> dict[str, float]:
    """
    For each fill, measure where the mid went N ticks later, signed by the
    direction we traded. Positive = the market moved our way = we were on
    the right side. Persistently negative = informed flow is running us over.
    Expressed in basis points so it's comparable across instruments.
    """
    out: dict[str, float] = {}
    for label, h in horizons_ticks.items():
        vals = []
        for mf in marked:
            j = mf.idx + h
            if j >= len(books):
                continue
            future_mid = books[j].mid
            direction = 1.0 if mf.fill.side == "buy" else -1.0
            move_bps = direction * (future_mid - mf.mid_at_fill) / mf.mid_at_fill * 10_000
            vals.append(move_bps)
        out[label] = round(statistics.mean(vals), 4) if vals else 0.0
    return out


def _sharpe(equity: list[float], dt_sec: float) -> float:
    if len(equity) < 3:
        return 0.0
    rets = [equity[i] - equity[i - 1] for i in range(1, len(equity))]
    sd = statistics.pstdev(rets)
    if sd < 1e-12:
        return 0.0
    periods_per_year = (365 * 24 * 3600) / max(dt_sec, 1e-9)
    return (statistics.mean(rets) / sd) * math.sqrt(periods_per_year)


def _max_dd(equity: list[float]) -> float:
    peak, dd = -float("inf"), 0.0
    for v in equity:
        peak = max(peak, v)
        dd = min(dd, v - peak)
    return abs(dd)


# ============================================================
# REPLAY ENGINE
# ============================================================

def replay(cfg: MarketConfig,
           data: list[tuple[BookState, float]],
           maker_fee_bps: float = -0.5,
           oms_cfg: Optional[OMSConfig] = None,
           cooldown_ticks: int = 12,
           verbose: bool = False) -> ReplayResult:
    """
    Run the full stack — engine, OMS, paper venue with queue modeling —
    over a tick series. Returns diagnostics, not just a PnL number.
    """
    books = [b for b, _ in data]

    SIM = {"t": books[0].timestamp if books else time.time()}
    adapter = PaperAdapter(maker_fee_bps=maker_fee_bps, tick_size=cfg.tick_size)
    mm = MarketMaker(cfg)
    oms = OrderManager(adapter, cfg.symbol, cfg.tick_size,
                       cfg=oms_cfg, clock=lambda: SIM["t"])

    state = {"fees": 0.0, "spread_pnl": 0.0, "idx": 0}
    marked: list[MarkedFill] = []

    def on_fill(f: Fill):
        state["fees"] += f.fee
        i = state["idx"]
        mid = books[i].mid
        marked.append(MarkedFill(fill=f, mid_at_fill=mid, idx=i))
        # Theoretical edge: distance from mid at the moment of the fill
        edge = (mid - f.price) if f.side == "buy" else (f.price - mid)
        state["spread_pnl"] += edge * f.size
        mm.on_fill(f.side, f.price, f.size)

    oms.on_fill(on_fill)

    equity: list[float] = []
    cooldown = 0
    quoted_ticks = 0

    for i, (book, volume) in enumerate(data):
        state["idx"] = i
        SIM["t"] = book.timestamp

        adapter.on_market_data(book, traded_volume=volume)
        oms.pump_fills()

        if cooldown > 0:
            cooldown -= 1
            mm.vol.update(book.mid, book.timestamp)
            oms.sync(None)
            equity.append(mm.inv.total_pnl(book.mid) - state["fees"])
            continue

        q = mm.compute_quote(book)

        if q is None and mm.halted:
            if "loss limit" in mm.halt_reason:
                if verbose:
                    print(f"[{i}] TERMINAL — {mm.halt_reason}")
                oms.sync(None)
                equity.append(mm.inv.total_pnl(book.mid) - state["fees"])
                break
            if verbose:
                print(f"[{i}] stand down — {mm.halt_reason}")
            mm.reset_halt()
            cooldown = cooldown_ticks

        oms.sync(q)

        if len(oms.resting) == 2:
            quoted_ticks += 1

        equity.append(mm.inv.total_pnl(book.mid) - state["fees"])

    oms.shutdown()

    n = max(len(equity), 1)
    final_mid = books[min(state["idx"], len(books) - 1)].mid
    total = mm.inv.total_pnl(final_mid) - state["fees"]

    dt = 0.25
    if len(books) > 1:
        span = books[-1].timestamp - books[0].timestamp
        if span > 0:
            dt = span / len(books)

    return ReplayResult(
        symbol=cfg.symbol,
        ticks=len(data),
        fills=len(marked),
        total_pnl=round(total, 2),
        spread_pnl=round(state["spread_pnl"], 2),
        inventory_pnl=round(total - state["spread_pnl"] + state["fees"], 2),
        fees=round(-state["fees"], 4),
        final_position=round(mm.inv.position, 6),
        sharpe=round(_sharpe(equity, dt), 2),
        max_drawdown=round(_max_dd(equity), 2),
        quote_uptime=quoted_ticks / n,
        markouts=_markout(marked, books, {
            "1 tick": 1, "4 ticks": 4, "20 ticks": 20, "120 ticks": 120,
        }),
        equity_curve=equity,
        oms_stats=oms.stats(),
    )


# ============================================================
# PARAMETER SWEEP
# ============================================================

def sweep_requote_threshold(cfg: MarketConfig,
                            data: list[tuple[BookState, float]],
                            values: list[float]) -> list[tuple[float, ReplayResult]]:
    """
    The requote threshold is the highest-leverage parameter. Too tight and
    you churn queue position; too loose and you get run over by stale quotes.
    Sweep it and look at markouts, not just PnL.
    """
    results = []
    for v in values:
        oms_cfg = OMSConfig(requote_threshold_bps=v)
        results.append((v, replay(cfg, data, oms_cfg=oms_cfg)))
    return results


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("Generating synthetic data with autocorrelated order flow...\n")
    data = synthetic_ticks(
        n=8000, start_px=95_000.0, vol_bps=6.0,
        spread_bps=9.0, flow_autocorr=0.7, seed=11,
    )

    res = replay(CRYPTO, data, maker_fee_bps=-0.5)
    print(res.report())

    print("\n\nREQUOTE THRESHOLD SWEEP")
    print("-" * 64)
    print(f"{'bps':>6} {'PnL':>10} {'fills':>7} {'uptime':>8} {'hold':>7} {'mo@20':>9}")
    print("-" * 64)
    for v, r in sweep_requote_threshold(CRYPTO, data, [0.5, 1.5, 3.0, 6.0, 12.0]):
        print(f"{v:>6.1f} {r.total_pnl:>10,.2f} {r.fills:>7} "
              f"{r.quote_uptime:>7.1%} {r.oms_stats['hold_rate']:>7.2f} "
              f"{r.markouts.get('20 ticks', 0):>+9.3f}")
    print("-" * 64)
    print("\nRead the markout column, not the PnL column. Negative markouts")
    print("mean the fills you're getting are the ones that hurt you.")
