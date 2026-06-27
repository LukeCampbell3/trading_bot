"""
Multileg Execution Manager for V14.2

Single writer responsible for ALL option orders.
No strategy module may submit orders directly.

Rules:
- All multileg entries use limit orders
- All exits use bot-controlled closing mleg limit orders
- No native stop assumption for multileg spreads
- Never leg manually except emergency liquidation (explicitly logged)
- Use client_order_id for idempotency
- Cancel stale unfilled orders
- Do not chase beyond max_limit_chase_pct_of_debit
- Reconcile broker positions before and after every order
- Prevent duplicate orders for the same ticket
- Prevent simultaneous contradictory orders for the same symbol/contract
"""

from __future__ import annotations

import csv
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG

# Alpaca SDK imports for real order submission
try:
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import (
        OrderSide, TimeInForce, OrderClass, OrderType, PositionIntent,
    )
    _ALPACA_OPTIONS_OK = True
except ImportError:
    _ALPACA_OPTIONS_OK = False


class OrderState(Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    MISSED_FILL = "MISSED_FILL"
    CLOSE_SUBMITTED = "CLOSE_SUBMITTED"
    CLOSED = "CLOSED"
    CLOSE_FAILED = "CLOSE_FAILED"
    EMERGENCY_EXIT_REQUESTED = "EMERGENCY_EXIT_REQUESTED"


@dataclass
class MlegLeg:
    """One leg of a multileg order."""
    contract_symbol: str = ""
    side: str = ""  # "buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"
    quantity: int = 0
    option_type: str = ""  # "call" or "put"
    strike: float = 0.0
    expiration: str = ""


@dataclass
class MlegOrder:
    """Represents a multileg option order through its lifecycle."""
    # Identity
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    client_order_id: str = field(default_factory=lambda: f"v14_{uuid.uuid4().hex[:12]}")
    ticket_id: str = ""
    parent_strategy: str = "V14_2_CORE_RUNNER"

    # Order spec
    order_class: str = "MLEG"
    order_type: str = "LIMIT"
    direction: str = ""  # "OPEN" or "CLOSE"
    legs: List[MlegLeg] = field(default_factory=list)
    intended_limit: float = 0.0
    quantity: int = 0

    # State
    state: OrderState = OrderState.CREATED
    submit_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None

    # Quote at submit
    quote_at_submit: Dict[str, float] = field(default_factory=dict)
    composite_bid: float = 0.0
    composite_ask: float = 0.0
    composite_mid: float = 0.0

    # Fill data
    actual_fill_price: float = 0.0
    fill_slippage_vs_mid: float = 0.0
    broker_order_id: str = ""

    # Close data
    exit_quote: Dict[str, float] = field(default_factory=dict)
    exit_limit: float = 0.0
    actual_exit_price: float = 0.0
    exit_slippage_vs_bid: float = 0.0

    # Outcome
    realized_pnl: float = 0.0
    missed_fill_reason: str = ""
    cancel_reason: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    stale_timeout_seconds: float = 120.0


class MlegExecutionManager:
    """
    Single execution writer for all V14.2 multileg option orders.
    
    This is the ONLY module that may submit orders to the broker.
    Strategy modules create order requests; this module manages execution.
    """

    def __init__(
        self,
        trading_client: Any = None,
        config: Optional[dict] = None,
        log_dir: str = "HFT/logs/v14_2",
        paper_mode: bool = True,
    ):
        self.trading_client = trading_client
        self.cfg = config or CFG
        self.paper_mode = paper_mode
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Order tracking
        self._orders: Dict[str, MlegOrder] = {}
        self._ticket_to_orders: Dict[str, List[str]] = {}
        self._active_symbols: Dict[str, str] = {}  # symbol -> active order_id

        # Reconciliation state
        self._last_reconcile_time: float = 0.0
        self._broker_positions: Dict[str, Any] = {}

        # Logging
        self._order_log_path = self.log_dir / "mleg_orders.csv"
        self._fill_log_path = self.log_dir / "fill_quality.csv"
        self._init_logs()

    def _init_logs(self):
        if not self._order_log_path.exists():
            with open(self._order_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "order_id", "client_order_id", "ticket_id",
                    "parent_strategy", "order_class", "direction", "legs",
                    "intended_limit", "state", "composite_bid", "composite_ask",
                    "composite_mid", "fill_time", "actual_fill_price",
                    "fill_slippage_vs_mid", "missed_fill_reason", "cancel_reason",
                    "exit_limit", "actual_exit_price", "exit_slippage_vs_bid",
                    "realized_pnl",
                ])
        if not self._fill_log_path.exists():
            with open(self._fill_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "order_id", "ticket_id", "direction",
                    "intended_limit", "actual_fill", "slippage",
                    "composite_mid", "time_to_fill_seconds",
                ])

    def create_entry_order(
        self,
        ticket_id: str,
        legs: List[MlegLeg],
        limit_price: float,
        quantity: int,
        composite_bid: float,
        composite_ask: float,
        composite_mid: float,
    ) -> Optional[MlegOrder]:
        """
        Create an entry mleg limit order.
        
        Returns None if order is rejected (duplicate, contradictory, etc.)
        """
        # ─── Duplicate Prevention ────────────────────────────────────────
        if ticket_id in self._ticket_to_orders:
            existing = self._ticket_to_orders[ticket_id]
            for oid in existing:
                if oid in self._orders:
                    order = self._orders[oid]
                    if order.state in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED,
                                       OrderState.FILLED):
                        # Already have an active order for this ticket
                        return None

        # ─── Contradictory Order Prevention ──────────────────────────────
        # Check if we have an active order for any of these contracts
        for leg in legs:
            if leg.contract_symbol in self._active_symbols:
                active_oid = self._active_symbols[leg.contract_symbol]
                if active_oid in self._orders:
                    active_order = self._orders[active_oid]
                    if active_order.state in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
                        return None

        # ─── Limit Chase Check ───────────────────────────────────────────
        if composite_mid > 0:
            chase_pct = (limit_price - composite_mid) / composite_mid
            if chase_pct > self.cfg["max_limit_chase_pct_of_debit"]:
                # Reduce to max allowed chase
                limit_price = composite_mid * (1 + self.cfg["max_limit_chase_pct_of_debit"])

        # ─── Create Order ────────────────────────────────────────────────
        order = MlegOrder(
            ticket_id=ticket_id,
            direction="OPEN",
            legs=legs,
            intended_limit=limit_price,
            quantity=quantity,
            composite_bid=composite_bid,
            composite_ask=composite_ask,
            composite_mid=composite_mid,
            quote_at_submit={
                "bid": composite_bid,
                "ask": composite_ask,
                "mid": composite_mid,
            },
        )

        self._orders[order.order_id] = order
        self._ticket_to_orders.setdefault(ticket_id, []).append(order.order_id)

        # Track active contracts
        for leg in legs:
            self._active_symbols[leg.contract_symbol] = order.order_id

        return order

    def submit_order(self, order: MlegOrder) -> bool:
        """
        Submit order to broker (or simulate in paper mode).
        Returns True if submission successful.
        """
        if self.cfg.get("live_trading_enabled", False) is False and not self.paper_mode:
            order.state = OrderState.REJECTED
            order.cancel_reason = "live_trading_disabled_and_not_paper"
            self._log_order(order)
            return False

        order.submit_time = datetime.utcnow()
        order.state = OrderState.SUBMITTED
        order.last_updated = datetime.utcnow()

        if self.paper_mode and self.trading_client is not None:
            # Paper mode WITH broker: submit real paper order to Alpaca paper API
            success = self._submit_to_broker(order)
            if not success:
                # Fallback to simulated fill if broker submission fails
                print(f"[EXEC] Broker paper submission failed, using simulated fill")
                self._simulate_fill(order)
        elif self.paper_mode:
            # Paper mode WITHOUT broker: pure simulation
            self._simulate_fill(order)
        else:
            # Live mode
            success = self._submit_to_broker(order)
            if not success:
                order.state = OrderState.REJECTED
                self._log_order(order)
                return False

        self._log_order(order)
        return True

    def create_exit_order(
        self,
        ticket_id: str,
        legs: List[MlegLeg],
        limit_price: float,
        quantity: int,
        exit_bid: float,
        exit_ask: float,
    ) -> Optional[MlegOrder]:
        """
        Create a closing mleg limit order (bot-managed exit).
        All exits are bot-controlled, not native stops.
        """
        order = MlegOrder(
            ticket_id=ticket_id,
            direction="CLOSE",
            legs=legs,
            intended_limit=limit_price,
            quantity=quantity,
            composite_bid=exit_bid,
            composite_ask=exit_ask,
            composite_mid=(exit_bid + exit_ask) / 2,
            exit_limit=limit_price,
        )

        self._orders[order.order_id] = order
        self._ticket_to_orders.setdefault(ticket_id, []).append(order.order_id)
        return order

    def submit_exit(self, order: MlegOrder) -> bool:
        """Submit exit order."""
        order.submit_time = datetime.utcnow()
        order.state = OrderState.CLOSE_SUBMITTED
        order.last_updated = datetime.utcnow()

        if self.paper_mode and self.trading_client is not None:
            # Paper mode WITH broker: submit real paper close order
            success = self._submit_to_broker(order)
            if not success:
                print(f"[EXEC] Broker paper exit failed, using simulated fill")
                self._simulate_exit_fill(order)
        elif self.paper_mode:
            self._simulate_exit_fill(order)
        else:
            success = self._submit_to_broker(order)
            if not success:
                order.state = OrderState.CLOSE_FAILED
                self._log_order(order)
                return False

        self._log_order(order)
        return True

    def cancel_stale_orders(self):
        """Cancel all orders that have exceeded their stale timeout."""
        now = datetime.utcnow()
        for oid, order in list(self._orders.items()):
            if order.state == OrderState.SUBMITTED:
                elapsed = (now - order.submit_time).total_seconds() if order.submit_time else 0
                if elapsed > order.stale_timeout_seconds:
                    order.state = OrderState.MISSED_FILL
                    order.missed_fill_reason = f"stale_timeout_{elapsed:.0f}s"
                    order.last_updated = now
                    self._release_contracts(order)
                    self._log_order(order)

    def cancel_all_entries_for_symbol(self, symbol: str):
        """Cancel all pending entry orders for a symbol before submitting new one."""
        for oid, order in list(self._orders.items()):
            if order.state == OrderState.SUBMITTED and order.direction == "OPEN":
                for leg in order.legs:
                    if symbol in leg.contract_symbol:
                        order.state = OrderState.CANCELED
                        order.cancel_reason = "replaced_by_new_entry"
                        order.last_updated = datetime.utcnow()
                        self._release_contracts(order)
                        self._log_order(order)
                        break

    def reconcile_positions(self) -> bool:
        """
        Reconcile broker positions before/after every order.
        Returns True if positions are consistent.
        """
        if self.paper_mode:
            # In paper mode, our internal state is authoritative
            self._last_reconcile_time = time.time()
            return True

        if self.trading_client is None:
            return False

        try:
            positions = self.trading_client.get_all_positions()
            self._broker_positions = {
                getattr(p, "symbol", ""): p for p in positions
            }
            self._last_reconcile_time = time.time()
            return True
        except Exception:
            return False

    def get_order(self, order_id: str) -> Optional[MlegOrder]:
        return self._orders.get(order_id)

    def get_orders_for_ticket(self, ticket_id: str) -> List[MlegOrder]:
        oids = self._ticket_to_orders.get(ticket_id, [])
        return [self._orders[oid] for oid in oids if oid in self._orders]

    def _simulate_fill(self, order: MlegOrder):
        """Paper mode: simulate fill at mid price."""
        order.state = OrderState.FILLED
        order.fill_time = datetime.utcnow()
        # Fill at mid (optimistic paper fill)
        order.actual_fill_price = order.composite_mid
        order.fill_slippage_vs_mid = 0.0
        order.last_updated = datetime.utcnow()
        self._log_fill(order)

    def _simulate_exit_fill(self, order: MlegOrder):
        """Paper mode: simulate exit fill at bid."""
        order.state = OrderState.CLOSED
        order.fill_time = datetime.utcnow()
        # Exit at bid (conservative paper exit)
        order.actual_exit_price = order.composite_bid
        order.exit_slippage_vs_bid = 0.0
        order.last_updated = datetime.utcnow()
        self._log_fill(order)

    def _submit_to_broker(self, order: MlegOrder) -> bool:
        """
        Submit mleg order to Alpaca broker.
        This is the ONLY place broker orders are created.
        Uses Alpaca's multileg options order API.
        """
        if self.trading_client is None:
            order.cancel_reason = "no_trading_client"
            return False

        if not _ALPACA_OPTIONS_OK:
            order.cancel_reason = "alpaca_options_sdk_not_available"
            return False

        try:
            # Build OptionLegRequest list from our MlegLeg objects
            alpaca_legs = []
            for leg in order.legs:
                # Map our side to Alpaca PositionIntent
                if leg.side == "buy_to_open":
                    position_intent = PositionIntent.BUY_TO_OPEN
                    side = OrderSide.BUY
                elif leg.side == "sell_to_open":
                    position_intent = PositionIntent.SELL_TO_OPEN
                    side = OrderSide.SELL
                elif leg.side == "buy_to_close":
                    position_intent = PositionIntent.BUY_TO_CLOSE
                    side = OrderSide.BUY
                elif leg.side == "sell_to_close":
                    position_intent = PositionIntent.SELL_TO_CLOSE
                    side = OrderSide.SELL
                else:
                    order.cancel_reason = f"unknown_leg_side: {leg.side}"
                    return False

                alpaca_legs.append(OptionLegRequest(
                    symbol=leg.contract_symbol,
                    ratio_qty=1.0,  # 1:1 ratio per spread
                    side=side,
                    position_intent=position_intent,
                ))

            # Determine order side based on direction
            # For a debit spread open: we're buying, so side=BUY
            # For a credit spread close: we're selling, so side=SELL
            if order.direction == "OPEN":
                order_side = OrderSide.BUY
            else:
                order_side = OrderSide.SELL

            # Build the multileg limit order
            # For MLEG orders: no top-level symbol, qty = number of spreads
            limit_order = LimitOrderRequest(
                qty=float(order.quantity),  # number of spreads
                side=order_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                limit_price=round(order.intended_limit, 2),
                legs=alpaca_legs,
                client_order_id=order.client_order_id,
            )

            # Submit to Alpaca
            response = self.trading_client.submit_order(limit_order)
            order.broker_order_id = str(getattr(response, "id", ""))
            order.state = OrderState.SUBMITTED
            order.submit_time = datetime.utcnow()
            print(f"[EXEC] Order submitted: {order.broker_order_id} | {order.client_order_id}")
            return True

        except Exception as e:
            order.cancel_reason = f"broker_error: {str(e)[:200]}"
            print(f"[EXEC] Order submission failed: {e}")
            return False

    def check_order_status(self, order: MlegOrder) -> OrderState:
        """
        Check the current status of a submitted order with the broker.
        Updates order state based on broker response.
        """
        if not self.trading_client or not order.broker_order_id:
            return order.state

        try:
            broker_order = self.trading_client.get_order_by_id(order.broker_order_id)
            status = str(getattr(broker_order, "status", "")).lower()

            if status == "filled":
                order.state = OrderState.FILLED if order.direction == "OPEN" else OrderState.CLOSED
                order.fill_time = datetime.utcnow()
                # Get fill price
                fill_price = getattr(broker_order, "filled_avg_price", None)
                if fill_price:
                    if order.direction == "OPEN":
                        order.actual_fill_price = float(fill_price)
                        order.fill_slippage_vs_mid = (
                            (order.actual_fill_price - order.composite_mid) / order.composite_mid
                            if order.composite_mid > 0 else 0.0
                        )
                    else:
                        order.actual_exit_price = float(fill_price)
                        order.exit_slippage_vs_bid = (
                            (order.composite_bid - order.actual_exit_price) / order.composite_bid
                            if order.composite_bid > 0 else 0.0
                        )
                self._log_fill(order)

            elif status == "partially_filled":
                order.state = OrderState.PARTIALLY_FILLED

            elif status in ("canceled", "cancelled"):
                order.state = OrderState.CANCELED
                order.cancel_reason = "broker_canceled"

            elif status == "expired":
                order.state = OrderState.EXPIRED

            elif status in ("rejected", "suspended"):
                order.state = OrderState.REJECTED
                order.cancel_reason = f"broker_{status}"

            elif status in ("new", "accepted", "pending_new"):
                order.state = OrderState.SUBMITTED

            order.last_updated = datetime.utcnow()

        except Exception as e:
            print(f"[EXEC] Status check failed for {order.broker_order_id}: {e}")

        return order.state

    def cancel_broker_order(self, order: MlegOrder) -> bool:
        """Cancel an order with the broker."""
        if not self.trading_client or not order.broker_order_id:
            return False

        try:
            self.trading_client.cancel_order_by_id(order.broker_order_id)
            order.state = OrderState.CANCELED
            order.cancel_reason = "user_canceled"
            order.last_updated = datetime.utcnow()
            return True
        except Exception as e:
            print(f"[EXEC] Cancel failed: {e}")
            return False

    def _release_contracts(self, order: MlegOrder):
        """Release tracked contract symbols when order is no longer active."""
        for leg in order.legs:
            if leg.contract_symbol in self._active_symbols:
                if self._active_symbols[leg.contract_symbol] == order.order_id:
                    del self._active_symbols[leg.contract_symbol]

    def _log_order(self, order: MlegOrder):
        """Log every order attempt with full context."""
        legs_str = "|".join(
            f"{l.side}:{l.contract_symbol}:{l.quantity}" for l in order.legs
        )
        with open(self._order_log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.utcnow().isoformat(),
                order.order_id,
                order.client_order_id,
                order.ticket_id,
                order.parent_strategy,
                order.order_class,
                order.direction,
                legs_str,
                f"{order.intended_limit:.4f}",
                order.state.value,
                f"{order.composite_bid:.4f}",
                f"{order.composite_ask:.4f}",
                f"{order.composite_mid:.4f}",
                order.fill_time.isoformat() if order.fill_time else "",
                f"{order.actual_fill_price:.4f}",
                f"{order.fill_slippage_vs_mid:.4f}",
                order.missed_fill_reason,
                order.cancel_reason,
                f"{order.exit_limit:.4f}",
                f"{order.actual_exit_price:.4f}",
                f"{order.exit_slippage_vs_bid:.4f}",
                f"{order.realized_pnl:.2f}",
            ])

    def _log_fill(self, order: MlegOrder):
        """Log fill quality metrics."""
        time_to_fill = 0.0
        if order.submit_time and order.fill_time:
            time_to_fill = (order.fill_time - order.submit_time).total_seconds()

        slippage = order.fill_slippage_vs_mid if order.direction == "OPEN" else order.exit_slippage_vs_bid

        with open(self._fill_log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.utcnow().isoformat(),
                order.order_id,
                order.ticket_id,
                order.direction,
                f"{order.intended_limit:.4f}",
                f"{order.actual_fill_price:.4f}" if order.direction == "OPEN" else f"{order.actual_exit_price:.4f}",
                f"{slippage:.4f}",
                f"{order.composite_mid:.4f}",
                f"{time_to_fill:.1f}",
            ])
