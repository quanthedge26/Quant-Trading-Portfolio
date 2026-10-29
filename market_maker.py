"""
Market Making Engine — Reference Implementation
================================================

Core quoting logic used by market makers, based on the Avellaneda-Stoikov
optimal market making framework (2008) with practical extensions for
inventory management and adverse selection protection.

This is the SAME math Citadel/Virtu/Jane Street use. What they have that
this doesn't: microsecond execution, co-location, and proprietary order
flow prediction. The quoting logic itself is public academic work.

PAPER TRADE THIS FIRST. Market making loses money fast when mispriced.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import math
import time


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class MarketConfig:
    """Per-market parameters. These are what you tune."""

    symbol: str
    venue: str

    # --- Position limits ---
    max_position: float          # max absolute inventory (contracts/coins)
    quote_size: float            # size posted on each side

    # --- Spread parameters ---
    min_spread_bps: float        # never quote tighter than this
    base_spread_bps: float       # baseline half-spread in basis points
    tick_size: float             # minimum price increment

    # --- Risk model ---
    gamma: float                 # inventory risk aversion (higher = skew harder)
    kappa: float                 # order book liquidity/fill intensity
    vol_lookback: int = 60       # seconds of returns for vol estimate

    # Cap the model's half-spread at this multiple of the OBSERVED book
    # half-spread. Without this, A-S's inventory term (which is in price
    # units) produces spreads hundreds of bps wide on high-priced assets,
    # and you quote so far outside the market you never fill. Real desks
    # always quote relative to the touch.
    max_spread_mult: float = 1.5

    # --- Safety ---
    max_daily_loss: float = 1000.0
    adverse_move_bps: float = 30.0   # pull quotes if mid moves this fast
    quote_refresh_ms: int = 500      # requote cadence
    inventory_flatten_pct: float = 0.9  # flatten aggressively above this % of max

    # Contracts trade in whole lots; crypto does not.
    integer_size: bool = False


# ------------------------------------------------------------
# Three market presets
# ------------------------------------------------------------

CRYPTO = MarketConfig(
    symbol="BTC-USD",
    venue="coinbase",
    max_position=0.5,            # 0.5 BTC
    quote_size=0.01,
    min_spread_bps=4.0,          # crypto spreads are wide — real edge here
    base_spread_bps=8.0,
    tick_size=0.01,
    gamma=0.15,
    kappa=1.2,
    max_daily_loss=500.0,
    adverse_move_bps=40.0,       # crypto is volatile, wider tolerance
    quote_refresh_ms=250,
)

FUTURES = MarketConfig(
    symbol="MNQ",                # Micro E-mini Nasdaq
    venue="ibkr",
    max_position=10,             # 10 contracts
    quote_size=1,
    min_spread_bps=0.8,          # futures spreads are TIGHT
    base_spread_bps=1.5,
    tick_size=0.25,              # MNQ tick = 0.25 index points
    gamma=0.35,                  # tighter inventory control — leverage is real
    kappa=2.5,
    max_daily_loss=750.0,
    adverse_move_bps=12.0,       # pull fast, you're competing with HFT
    quote_refresh_ms=100,
    integer_size=True,
)

EQUITY_OPTIONS = MarketConfig(
    symbol="AMD  260320C00200000",
    venue="ibkr",
    max_position=25,             # 25 contracts
    quote_size=2,
    min_spread_bps=60.0,         # options spreads are very wide
    base_spread_bps=120.0,
    tick_size=0.01,
    gamma=0.5,                   # options carry Greeks risk — skew hard
    kappa=12.0,                  # kappa has units of 1/price — a $4.50
                                 # option needs a far larger kappa than BTC
    max_daily_loss=1000.0,
    adverse_move_bps=200.0,      # options mids are noisy; don't over-halt
    quote_refresh_ms=1000,       # no speed race here
    integer_size=True,
)


# ============================================================
# STATE
# ============================================================

@dataclass
class BookState:
    """Live market snapshot from the venue."""
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def microprice(self) -> float:
        """
        Size-weighted mid. Leans toward the side with LESS size, because
        that side is more likely to get lifted next. This is a real edge —
        it front-runs the mid's own movement.
        """
        total = self.bid_size + self.ask_size
        if total <= 0:
            return self.mid
        return (self.bid * self.ask_size + self.ask * self.bid_size) / total


@dataclass
class Inventory:
    position: float = 0.0
    avg_entry: float = 0.0
    realized_pnl: float = 0.0

    def on_fill(self, side: Literal["buy", "sell"], price: float, qty: float):
        signed = qty if side == "buy" else -qty
        new_pos = self.position + signed

        # Closing or flipping — realize PnL on the closed portion
        if self.position != 0 and (self.position > 0) != (signed > 0):
            closed = min(abs(signed), abs(self.position))
            direction = 1 if self.position > 0 else -1
            self.realized_pnl += direction * closed * (price - self.avg_entry)

            if abs(signed) > abs(self.position):
                self.avg_entry = price       # flipped sides
            elif new_pos == 0:
                self.avg_entry = 0.0
        else:
            # Adding to position — weighted average entry
            if new_pos != 0:
                self.avg_entry = (
                    self.avg_entry * abs(self.position) + price * abs(signed)
                ) / abs(new_pos)

        self.position = new_pos

    def unrealized(self, mark: float) -> float:
        if self.position == 0:
            return 0.0
        return self.position * (mark - self.avg_entry)

    def total_pnl(self, mark: float) -> float:
        return self.realized_pnl + self.unrealized(mark)


# ============================================================
# VOLATILITY ESTIMATOR
# ============================================================

class RollingVol:
    """
    EWMA realized volatility from mid-price returns.
    Vol drives spread width — wider vol, wider quotes.
    """

    def __init__(self, halflife_sec: float = 30.0):
        self.lam = math.exp(-math.log(2) / halflife_sec)
        self.var = 0.0
        self.last_mid: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.samples = 0

    def update(self, mid: float, ts: float) -> float:
        if self.last_mid is not None and self.last_ts is not None:
            dt = max(ts - self.last_ts, 1e-6)
            ret = math.log(mid / self.last_mid)
            # normalize to per-second variance
            inst_var = (ret * ret) / dt
            self.var = self.lam * self.var + (1 - self.lam) * inst_var
            self.samples += 1

        self.last_mid = mid
        self.last_ts = ts
        return self.sigma

    @property
    def sigma(self) -> float:
        """Per-second volatility. Returns a floor until warmed up."""
        if self.samples < 10:
            return 1e-4
        return math.sqrt(max(self.var, 1e-12))

    @property
    def ready(self) -> bool:
        return self.samples >= 10


# ============================================================
# THE QUOTING ENGINE
# ============================================================

@dataclass
class Quote:
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    reason: str = "normal"

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class MarketMaker:
    """
    Avellaneda-Stoikov optimal quoting with inventory skew.

    THE CORE IDEA
    -------------
    You do NOT quote symmetrically around the mid. You quote around your
    *reservation price* — the price at which you're indifferent to your
    current inventory.

        r = s - q * gamma * sigma^2 * T

    If you're long, r < mid, so both quotes shift DOWN. Your ask becomes
    more attractive, your bid less attractive, and the market naturally
    flattens you out. This is the entire game — spread capture is easy,
    not blowing up on inventory is hard.

    The optimal total spread:

        delta = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)

    First term = inventory risk premium. Second = the fill-intensity
    optimum given how aggressively the book trades against you.
    """

    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.inv = Inventory()
        self.vol = RollingVol(halflife_sec=cfg.vol_lookback / 2)
        self.session_start = time.time()
        self.halted = False
        self.halt_reason = ""
        self._mid_history: list[tuple[float, float]] = []

    # --------------------------------------------------------
    # Risk gates — checked before every quote
    # --------------------------------------------------------

    def _check_risk(self, book: BookState) -> Optional[str]:
        cfg = self.cfg

        pnl = self.inv.total_pnl(book.mid)
        if pnl <= -cfg.max_daily_loss:
            return f"daily loss limit hit ({pnl:.2f})"

        if abs(self.inv.position) > cfg.max_position:
            return f"position over limit ({self.inv.position})"

        # Adverse selection: mid ripped away from us recently
        now = book.timestamp
        self._mid_history.append((now, book.mid))
        self._mid_history = [(t, m) for t, m in self._mid_history if now - t <= 2.0]

        if len(self._mid_history) >= 2:
            oldest = self._mid_history[0][1]
            move_bps = abs(book.mid - oldest) / oldest * 10_000
            if move_bps > cfg.adverse_move_bps:
                return f"adverse move {move_bps:.1f}bps in 2s"

        # Crossed or locked book — never quote into this
        if book.ask <= book.bid:
            return "crossed/locked book"

        return None

    # --------------------------------------------------------
    # Reservation price + optimal spread
    # --------------------------------------------------------

    def reservation_price(self, mid: float, sigma_px: float, T: float) -> float:
        """
        sigma_px is volatility in PRICE units (dollars), not returns.
        Skew is scaled by max_position so the shift at full inventory is
        a meaningful fraction of the spread, not a rounding error.
        """
        q = self.inv.position / max(self.cfg.max_position, 1e-9)  # -1..+1
        skew = q * self.cfg.gamma * (sigma_px ** 2) * T
        return mid - skew

    def optimal_half_spread(self, sigma_px: float, T: float) -> float:
        """
        A-S optimal spread, in price units:
            delta = gamma*sigma^2*T + (2/gamma)*ln(1 + gamma/kappa)
        kappa carries units of 1/price, so the intensity term already
        comes out in dollars. Do NOT rescale it by the asset price.
        """
        g, k = self.cfg.gamma, self.cfg.kappa
        inventory_term = g * (sigma_px ** 2) * T
        intensity_term = (2.0 / g) * math.log(1.0 + g / k)
        return 0.5 * (inventory_term + intensity_term)

    # --------------------------------------------------------
    # Main entry point
    # --------------------------------------------------------

    def compute_quote(self, book: BookState, horizon_sec: float = 60.0) -> Optional[Quote]:
        cfg = self.cfg

        if self.halted:
            return None

        reason = self._check_risk(book)
        if reason:
            self.halted = True
            self.halt_reason = reason
            return None

        sigma_ret = self.vol.update(book.mid, book.timestamp)

        # Anchor on microprice, not mid — it leads the mid
        anchor = book.microprice

        # A-S is formulated in absolute price units
        sigma_px = sigma_ret * anchor

        # Normalize horizon; T shrinks toward session end in real deployments
        T = horizon_sec / 60.0

        r = self.reservation_price(anchor, sigma_px, T)
        half_model = self.optimal_half_spread(sigma_px, T)

        # Clamp to a tradeable band around the observed book. The floor keeps
        # us from quoting inside our own edge; the ceiling keeps the A-S
        # inventory term from pushing us so far out we never get filled.
        book_half = max(book.spread / 2.0, cfg.tick_size)
        half_price = min(
            max(
                half_model,
                cfg.min_spread_bps / 10_000 * anchor / 2,
                cfg.tick_size,
            ),
            book_half * cfg.max_spread_mult,
        )

        # The skew must also be bounded — an unclamped reservation price can
        # shift further than the spread itself and invert the quote.
        max_skew = book_half * cfg.max_spread_mult
        r = anchor + max(-max_skew, min(max_skew, r - anchor))

        bid = r - half_price
        ask = r + half_price

        # --- Never quote through the existing book ---
        bid = min(bid, book.bid + cfg.tick_size)
        ask = max(ask, book.ask - cfg.tick_size)

        bid = self._round_to_tick(bid, "down")
        ask = self._round_to_tick(ask, "up")

        if ask - bid < cfg.tick_size:
            ask = bid + cfg.tick_size

        # --- Inventory-aware sizing ---
        bid_sz, ask_sz, note = self._size_quotes()

        return Quote(bid=bid, ask=ask, bid_size=bid_sz, ask_size=ask_sz, reason=note)

    def _size_quotes(self) -> tuple[float, float, str]:
        """
        Skew SIZE as well as price. Long inventory → smaller bid, bigger ask.
        Above the flatten threshold, stop quoting the adding side entirely.
        """
        cfg = self.cfg
        base = cfg.quote_size
        util = self.inv.position / max(cfg.max_position, 1e-9)   # -1..+1

        if util >= cfg.inventory_flatten_pct:
            return 0.0, base * 2, "flattening long"
        if util <= -cfg.inventory_flatten_pct:
            return base * 2, 0.0, "flattening short"

        bid_sz = base * max(0.0, 1.0 - util)
        ask_sz = base * max(0.0, 1.0 + util)

        if cfg.integer_size:
            # Round to whole lots. Never round a live side down to zero —
            # that silently stops quoting instead of just quoting small.
            bid_sz = max(1.0, round(bid_sz)) if bid_sz > 0 else 0.0
            ask_sz = max(1.0, round(ask_sz)) if ask_sz > 0 else 0.0

        note = "normal"
        if abs(util) > 0.5:
            note = "skewed"
        return bid_sz, ask_sz, note

    def _round_to_tick(self, price: float, direction: str) -> float:
        t = self.cfg.tick_size
        return math.floor(price / t) * t if direction == "down" else math.ceil(price / t) * t

    # --------------------------------------------------------
    # Fill handling
    # --------------------------------------------------------

    def on_fill(self, side: Literal["buy", "sell"], price: float, qty: float):
        self.inv.on_fill(side, price, qty)

    def reset_halt(self):
        self.halted = False
        self.halt_reason = ""
        self._mid_history.clear()

    def status(self, mark: float) -> dict:
        return {
            "symbol": self.cfg.symbol,
            "position": round(self.inv.position, 6),
            "avg_entry": round(self.inv.avg_entry, 4),
            "realized": round(self.inv.realized_pnl, 2),
            "unrealized": round(self.inv.unrealized(mark), 2),
            "total_pnl": round(self.inv.total_pnl(mark), 2),
            "sigma_bps": round(self.vol.sigma * 10_000, 3),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }


# ============================================================
# BACKTEST / SIMULATION HARNESS
# ============================================================

def simulate(cfg: MarketConfig, ticks: list[BookState], verbose: bool = True):
    """
    Naive fill model: you get filled when the market trades through your
    quote. Real backtests need queue position modeling — you are LAST in
    line at your price level, so you only fill after everyone ahead of you.
    This sim is optimistic. Treat results as an upper bound.
    """
    mm = MarketMaker(cfg)
    fills = 0
    halts = 0
    cooldown_until = -1
    COOLDOWN_TICKS = 10   # stand down after an adverse move, then re-enter

    for i, book in enumerate(ticks):
        # Keep the vol estimator warm even while stood down
        if i < cooldown_until:
            mm.vol.update(book.mid, book.timestamp)
            continue

        q = mm.compute_quote(book)
        if q is None:
            if mm.halted:
                halts += 1
                if verbose:
                    print(f"[{i}] HALTED — {mm.halt_reason}")
                # A loss-limit breach is terminal. Everything else cools down.
                if "loss limit" in mm.halt_reason:
                    break
                mm.reset_halt()
                cooldown_until = i + COOLDOWN_TICKS
            continue

        if i + 1 >= len(ticks):
            break
        nxt = ticks[i + 1]

        # Someone sold into our bid
        if nxt.bid <= q.bid and q.bid_size > 0:
            mm.on_fill("buy", q.bid, q.bid_size)
            fills += 1
        # Someone bought into our ask
        if nxt.ask >= q.ask and q.ask_size > 0:
            mm.on_fill("sell", q.ask, q.ask_size)
            fills += 1

    final = ticks[-1].mid
    st = mm.status(final)
    st["fills"] = fills
    st["halts"] = halts
    return st


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    import random

    def synth_ticks(n: int, start: float, vol_bps: float, spread_bps: float):
        out, px = [], start
        for _ in range(n):
            px *= math.exp(random.gauss(0, vol_bps / 10_000))
            h = px * spread_bps / 10_000 / 2
            out.append(BookState(
                bid=px - h, ask=px + h,
                bid_size=random.uniform(1, 10),
                ask_size=random.uniform(1, 10),
                timestamp=time.time() + len(out) * 0.5,
            ))
        return out

    random.seed(42)

    print("=" * 62)
    print("MARKET MAKING SIMULATION — three venues")
    print("=" * 62)

    for cfg, px, vol_bps in [
        (CRYPTO, 95_000.0, 8.0),
        (FUTURES, 21_500.0, 2.0),
        (EQUITY_OPTIONS, 4.50, 15.0),
    ]:
        ticks = synth_ticks(2000, px, vol_bps, cfg.base_spread_bps * 2)
        res = simulate(cfg, ticks, verbose=False)
        print(f"\n{cfg.symbol}  ({cfg.venue})")
        print("-" * 62)
        for k, v in res.items():
            print(f"  {k:<14} {v}")
