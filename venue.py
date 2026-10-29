"""
Venue Adapters + Order Manager
==============================

Connects the quoting engine in market_maker.py to a real (or paper) venue.

The quoting engine says "I want to be bid 100.25 for 5, offered 100.75 for 5."
This layer answers the harder question: given what's already resting in the
book under my account, what orders do I actually need to send RIGHT NOW?

The single most important behavior here is NOT requoting. Every cancel-replace:
  - loses your queue position (you go to the back of the line at that price)
  - burns rate limit
  - opens a window where you're quoting one-sided or not at all

So the manager holds a quote until the desired price moves by more than a
configurable threshold. Amateur bots requote on every tick and wonder why
they never get filled on the good side.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Protocol
from enum import Enum
import time
import uuid
import logging

log = logging.getLogger("venue")

Side = Literal["buy", "sell"]


# ============================================================
# ORDER MODEL
# ============================================================

class OrderState(Enum):
    PENDING_NEW = "pending_new"      # sent, not yet acked
    LIVE = "live"                    # resting on the book
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    FILLED = "filled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in (OrderState.CANCELLED, OrderState.FILLED, OrderState.REJECTED)

    @property
    def working(self) -> bool:
        """Order is consuming an exchange slot — counts against our position risk."""
        return self in (OrderState.PENDING_NEW, OrderState.LIVE, OrderState.PENDING_CANCEL)


@dataclass
class Order:
    client_id: str
    symbol: str
    side: Side
    price: float
    size: float
    state: OrderState = OrderState.PENDING_NEW
    venue_id: Optional[str] = None
    filled_qty: float = 0.0
    created_at: float = field(default_factory=time.time)
    acked_at: Optional[float] = None

    @property
    def remaining(self) -> float:
        return max(0.0, self.size - self.filled_qty)

    @property
    def age_sec(self) -> float:
        return time.time() - self.created_at


@dataclass
class Fill:
    client_id: str
    symbol: str
    side: Side
    price: float
    size: float
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)
    is_maker: bool = True


# ============================================================
# ADAPTER INTERFACE
# ============================================================

class VenueAdapter(Protocol):
    """
    What every venue must implement. Keep this surface small — venue-specific
    weirdness belongs inside the adapter, not leaking into the strategy.
    """

    def place(self, symbol: str, side: Side, price: float, size: float,
              client_id: str) -> Optional[str]: ...

    def cancel(self, order: Order) -> bool: ...

    def open_orders(self, symbol: str) -> list[Order]: ...

    def poll_fills(self) -> list[Fill]: ...

    @property
    def name(self) -> str: ...


# ============================================================
# PAPER ADAPTER — test everything here first
# ============================================================

class PaperAdapter:
    """
    Simulated venue with a QUEUE POSITION model, which is the thing most
    paper traders omit and the reason their live results collapse.

    When you post a limit order, you join the back of the queue at that price.
    You only fill after the volume ahead of you is consumed. That means:
      - You miss fills you'd have "gotten" in a naive backtest
      - You disproportionately fill when the market is moving against you
        (adverse selection), because that's when the queue actually clears

    Set realistic queue_ahead values from real book data if you have it.
    """

    def __init__(self, maker_fee_bps: float = -0.5, latency_ms: float = 20.0,
                 tick_size: float = 0.01):
        # Negative maker fee = rebate. Many venues pay you to post.
        self.maker_fee_bps = maker_fee_bps
        self.latency_ms = latency_ms
        # Needed to decide "is my order AT the touch". Exact float equality
        # fails here — quotes are rounded to tick and prices carry FP noise.
        self.tick_size = tick_size
        self._orders: dict[str, Order] = {}
        self._queue_ahead: dict[str, float] = {}
        self._pending_fills: list[Fill] = []
        self._last_book = None

    @property
    def name(self) -> str:
        return "paper"

    def place(self, symbol, side, price, size, client_id) -> Optional[str]:
        vid = f"paper-{uuid.uuid4().hex[:10]}"
        o = Order(client_id=client_id, symbol=symbol, side=side,
                  price=price, size=size, venue_id=vid)
        o.state = OrderState.LIVE
        o.acked_at = time.time()
        self._orders[client_id] = o

        # Queue position: assume the resting size at our price is ahead of us.
        tol = self.tick_size / 2
        if self._last_book is not None:
            b = self._last_book
            if side == "buy" and abs(price - b.bid) < tol:
                self._queue_ahead[client_id] = b.bid_size
            elif side == "sell" and abs(price - b.ask) < tol:
                self._queue_ahead[client_id] = b.ask_size
            else:
                # Improving the book — we're alone at this level, front of queue
                self._queue_ahead[client_id] = 0.0
        else:
            self._queue_ahead[client_id] = 0.0

        return vid

    def cancel(self, order: Order) -> bool:
        o = self._orders.get(order.client_id)
        if o is None or o.state.terminal:
            return False
        o.state = OrderState.CANCELLED
        self._queue_ahead.pop(order.client_id, None)
        return True

    def open_orders(self, symbol: str) -> list[Order]:
        return [o for o in self._orders.values()
                if o.symbol == symbol and o.state.working]

    def poll_fills(self) -> list[Fill]:
        out = self._pending_fills
        self._pending_fills = []
        return out

    # --- simulation driver ---

    def on_market_data(self, book, traded_volume: float = 0.0):
        """
        Feed this a BookState each tick. traded_volume is how much printed
        at the touch since the last call — that's what eats the queue.
        """
        self._last_book = book
        tol = self.tick_size / 2

        for cid, o in list(self._orders.items()):
            if o.state != OrderState.LIVE:
                continue

            # Crossed: our price is through the far side, we'd be a taker.
            # (post_only would reject this live; here we fill it as the
            # worst case, which is the honest assumption.)
            crossed = (o.side == "buy" and o.price >= book.ask) or \
                      (o.side == "sell" and o.price <= book.bid)

            # Queue-eligible: resting AT the touch or improving on it.
            # Exact equality is wrong — a quote one tick inside the book is
            # the new best price and is absolutely eligible to fill.
            eligible = (o.side == "buy" and o.price >= book.bid - tol) or \
                       (o.side == "sell" and o.price <= book.ask + tol)

            if crossed:
                # Market traded through our price — we fill regardless of queue
                self._fill(o, o.remaining)
            elif eligible and traded_volume > 0:
                ahead = self._queue_ahead.get(cid, 0.0)
                consumed = min(traded_volume, ahead)
                self._queue_ahead[cid] = ahead - consumed
                leftover = traded_volume - consumed
                if leftover > 0:
                    self._fill(o, min(leftover, o.remaining))

    def _fill(self, o: Order, qty: float):
        if qty <= 0:
            return
        o.filled_qty += qty
        fee = o.price * qty * (self.maker_fee_bps / 10_000)
        self._pending_fills.append(Fill(
            client_id=o.client_id, symbol=o.symbol, side=o.side,
            price=o.price, size=qty, fee=fee, is_maker=True,
        ))
        if o.remaining <= 1e-12:
            o.state = OrderState.FILLED
            self._queue_ahead.pop(o.client_id, None)


# ============================================================
# LIVE ADAPTER SKELETONS
# ============================================================

class CoinbaseAdapter:
    """
    Coinbase Advanced Trade. Fill in auth with your own API credentials.

    Notes that will bite you:
      - post_only=True is MANDATORY for market making. Without it a quote
        that crosses becomes a TAKER order: you pay the taker fee instead of
        earning the rebate, and you've just bought the top of a move.
      - Rate limits are per-key. Cancel-replace burns two calls.
      - WebSocket 'user' channel gives fills faster than REST polling.
        Poll REST only as a reconciliation backstop.
    """

    BASE = "https://api.coinbase.com/api/v3/brokerage"

    def __init__(self, api_key: str, api_secret: str, post_only: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.post_only = post_only
        self._session = None   # requests.Session() or httpx.Client()

    @property
    def name(self) -> str:
        return "coinbase"

    def place(self, symbol, side, price, size, client_id) -> Optional[str]:
        payload = {
            "client_order_id": client_id,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": str(price),
                    "post_only": self.post_only,   # never cross. non-negotiable.
                }
            },
        }
        raise NotImplementedError(
            "Wire up signed POST to /orders with your API credentials. "
            f"Payload shape ready: {payload}"
        )

    def cancel(self, order: Order) -> bool:
        raise NotImplementedError("POST /orders/batch_cancel with order_ids=[venue_id]")

    def open_orders(self, symbol: str) -> list[Order]:
        raise NotImplementedError("GET /orders/historical/batch?order_status=OPEN")

    def poll_fills(self) -> list[Fill]:
        raise NotImplementedError("GET /orders/historical/fills, or WS 'user' channel")


class IBKRAdapter:
    """
    Interactive Brokers via ib_insync. Works for both MNQ futures and
    equity options.

    Notes that will bite you:
      - TWS/Gateway must be running. There is no direct REST path.
      - Order IDs must be monotonically increasing per session.
      - Options: check that the contract actually has a two-sided market
        before quoting. Illiquid strikes will hand you inventory you
        cannot get out of.
      - Futures: use `orderType='LMT'` with `tif='DAY'`. Set outsideRth
        deliberately, not by accident.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host, self.port, self.ib_client_id = host, port, client_id
        self.ib = None   # ib_insync.IB()

    @property
    def name(self) -> str:
        return "ibkr"

    def connect(self):
        raise NotImplementedError(
            "from ib_insync import IB; self.ib = IB(); "
            "self.ib.connect(self.host, self.port, clientId=self.ib_client_id)"
        )

    def place(self, symbol, side, price, size, client_id) -> Optional[str]:
        raise NotImplementedError(
            "contract = Future('MNQ', '202603', 'CME')  # or Option(...)\n"
            "order = LimitOrder(side.upper(), size, price, tif='DAY')\n"
            "trade = self.ib.placeOrder(contract, order)"
        )

    def cancel(self, order: Order) -> bool:
        raise NotImplementedError("self.ib.cancelOrder(trade.order)")

    def open_orders(self, symbol: str) -> list[Order]:
        raise NotImplementedError("self.ib.openTrades()")

    def poll_fills(self) -> list[Fill]:
        raise NotImplementedError("self.ib.fills() or the execDetailsEvent handler")


# ============================================================
# ORDER MANAGER — the quote lifecycle
# ============================================================

@dataclass
class OMSConfig:
    # Requote only when the desired price moves more than BOTH thresholds.
    # THIS IS THE MOST IMPORTANT PARAMETER IN THE FILE.
    # 0 = requote constantly = you never hold queue position = you lose.
    #
    # Ticks alone don't scale: 1 tick is $0.01 on BTC (meaningless, requotes
    # every tick) and 0.25 index points on MNQ (significant). The bps floor
    # makes the threshold price-relative so one config works everywhere.
    requote_threshold_ticks: float = 1.0
    requote_threshold_bps: float = 1.5

    # Also requote if the resting order gets this stale, to avoid drift.
    max_quote_age_sec: float = 30.0

    # Requote if desired size differs from resting by more than this fraction.
    size_tolerance_pct: float = 0.35

    max_orders_per_sec: float = 8.0
    cancel_all_on_shutdown: bool = True


class OrderManager:
    """
    Reconciles desired quotes against resting orders.

    Call sync() with the quote your strategy wants. It works out the minimal
    set of cancels and places needed and executes them, subject to rate limits.
    """

    def __init__(self, adapter: VenueAdapter, symbol: str,
                 tick_size: float, cfg: OMSConfig | None = None,
                 clock: Callable[[], float] = time.time):
        self.adapter = adapter
        self.symbol = symbol
        self.tick_size = tick_size
        self.cfg = cfg or OMSConfig()

        # Injectable so backtests rate-limit against SIMULATED time.
        # Using wall-clock in a backtest throttles you to a handful of
        # orders across the entire run and hides the real behavior.
        self.clock = clock

        self.resting: dict[Side, Order] = {}
        self._call_times: list[float] = []
        self._fill_callbacks: list[Callable[[Fill], None]] = []

        self.orders_sent = 0
        self.cancels_sent = 0
        self.requotes_skipped = 0
        self.rate_limited = 0

    def on_fill(self, cb: Callable[[Fill], None]):
        self._fill_callbacks.append(cb)

    # --------------------------------------------------------

    def _rate_ok(self) -> bool:
        now = self.clock()
        self._call_times = [t for t in self._call_times if now - t < 1.0]
        if len(self._call_times) >= self.cfg.max_orders_per_sec:
            return False
        self._call_times.append(now)
        return True

    def _needs_requote(self, side: Side, price: float, size: float) -> bool:
        cur = self.resting.get(side)
        if cur is None or cur.state.terminal:
            return True

        move = abs(price - cur.price)
        min_move = max(
            self.cfg.requote_threshold_ticks * self.tick_size,
            self.cfg.requote_threshold_bps / 10_000 * cur.price,
        )
        if move >= min_move:
            return True

        if cur.remaining > 0:
            size_drift = abs(size - cur.remaining) / max(cur.remaining, 1e-9)
            if size_drift > self.cfg.size_tolerance_pct:
                return True

        if self.clock() - cur.created_at > self.cfg.max_quote_age_sec:
            return True

        self.requotes_skipped += 1
        return False

    def _cancel_side(self, side: Side):
        cur = self.resting.get(side)
        if cur is None or cur.state.terminal:
            self.resting.pop(side, None)
            return
        if self.adapter.cancel(cur):
            self.cancels_sent += 1
        self.resting.pop(side, None)

    def _place_side(self, side: Side, price: float, size: float):
        if size <= 0:
            return
        cid = f"mm-{side}-{uuid.uuid4().hex[:8]}"
        vid = self.adapter.place(self.symbol, side, price, size, cid)
        if vid is None:
            log.warning("place rejected: %s %s @ %s", side, size, price)
            return
        self.resting[side] = Order(
            client_id=cid, symbol=self.symbol, side=side,
            price=price, size=size, venue_id=vid, state=OrderState.LIVE,
            created_at=self.clock(),
        )
        self.orders_sent += 1

    # --------------------------------------------------------

    def sync(self, quote) -> dict:
        """
        quote: a Quote from MarketMaker.compute_quote(), or None to stand down.
        Returns a small action report.
        """
        actions = {"placed": [], "cancelled": [], "held": []}

        # Strategy halted or standing down — pull everything.
        if quote is None:
            for side in ("buy", "sell"):
                if side in self.resting:
                    self._cancel_side(side)
                    actions["cancelled"].append(side)
            return actions

        targets = {
            "buy": (quote.bid, quote.bid_size),
            "sell": (quote.ask, quote.ask_size),
        }

        for side, (price, size) in targets.items():
            if size <= 0:
                if side in self.resting:
                    self._cancel_side(side)
                    actions["cancelled"].append(side)
                continue

            if not self._needs_requote(side, price, size):
                actions["held"].append(side)
                continue

            if not self._rate_ok():
                self.rate_limited += 1
                actions["held"].append(side)
                continue

            if side in self.resting:
                self._cancel_side(side)
            self._place_side(side, price, size)
            actions["placed"].append((side, round(price, 6), size))

        return actions

    def pump_fills(self):
        """Drain fills from the venue and fan them out to subscribers."""
        for f in self.adapter.poll_fills():
            cur = self.resting.get(f.side)
            if cur and cur.client_id == f.client_id:
                cur.filled_qty += f.size
                if cur.remaining <= 1e-12:
                    cur.state = OrderState.FILLED
                    self.resting.pop(f.side, None)
            for cb in self._fill_callbacks:
                cb(f)

    def reconcile(self):
        """
        Run at startup. Any order resting at the venue that we don't know
        about is an orphan from a previous crashed session — kill it.
        Skipping this is how bots wake up with mystery positions.
        """
        known = {o.venue_id for o in self.resting.values()}
        orphans = [o for o in self.adapter.open_orders(self.symbol)
                   if o.venue_id not in known]
        for o in orphans:
            log.warning("cancelling orphan order %s", o.venue_id)
            self.adapter.cancel(o)
        return len(orphans)

    def shutdown(self):
        if self.cfg.cancel_all_on_shutdown:
            for side in list(self.resting.keys()):
                self._cancel_side(side)

    def stats(self) -> dict:
        total = self.orders_sent + self.requotes_skipped
        return {
            "orders_sent": self.orders_sent,
            "cancels_sent": self.cancels_sent,
            "requotes_skipped": self.requotes_skipped,
            "rate_limited": self.rate_limited,
            "hold_rate": round(self.requotes_skipped / max(total, 1), 3),
        }


# ============================================================
# DEMO — full loop wired together
# ============================================================

if __name__ == "__main__":
    import math, random
    from market_maker import MarketMaker, BookState, CRYPTO

    random.seed(7)
    logging.basicConfig(level=logging.WARNING)

    SIM = {"t": time.time()}
    sim_clock = lambda: SIM["t"]

    adapter = PaperAdapter(maker_fee_bps=-0.5)
    mm = MarketMaker(CRYPTO)
    oms = OrderManager(adapter, CRYPTO.symbol, CRYPTO.tick_size, clock=sim_clock)

    fees = 0.0

    def handle_fill(f: Fill):
        global fees
        mm.on_fill(f.side, f.price, f.size)
        fees += f.fee

    oms.on_fill(handle_fill)

    px = 95_000.0
    cooldown = 0

    for i in range(3000):
        SIM["t"] += 0.25          # 250ms per tick of simulated time
        px *= math.exp(random.gauss(0, 8 / 10_000))
        half = px * 8 / 10_000 / 2
        book = BookState(
            bid=round(px - half, 2), ask=round(px + half, 2),
            bid_size=random.uniform(0.5, 4.0),
            ask_size=random.uniform(0.5, 4.0),
            timestamp=SIM["t"],
        )

        adapter.on_market_data(book, traded_volume=random.uniform(0, 0.6))
        oms.pump_fills()

        if cooldown > 0:
            cooldown -= 1
            mm.vol.update(book.mid, book.timestamp)
            oms.sync(None)
            continue

        q = mm.compute_quote(book)
        if q is None and mm.halted:
            if "loss limit" in mm.halt_reason:
                print(f"[{i}] TERMINAL: {mm.halt_reason}")
                oms.sync(None)
                break
            mm.reset_halt()
            cooldown = 12

        oms.sync(q)

    oms.shutdown()

    print("=" * 60)
    print("PAPER SESSION — BTC-USD with queue-position fills")
    print("=" * 60)
    for k, v in mm.status(px).items():
        print(f"  {k:<14} {v}")
    print()
    for k, v in oms.stats().items():
        print(f"  {k:<14} {v}")
    print(f"  {'fees_earned':<14} {round(-fees, 4)}")
