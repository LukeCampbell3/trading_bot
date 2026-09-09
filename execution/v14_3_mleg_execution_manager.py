"""Strict broker-backed multileg execution for V14.3.

The legacy V14.2 manager intentionally supports an offline simulation fallback.
That behavior is useful for unit tests, but it is unsafe for communication
validation because an Alpaca failure can look like a successful simulated fill.

This manager fails closed whenever a broker client is supplied:
- broker submission failures are never converted into synthetic fills;
- paper accounts are reconciled against Alpaca just like live accounts;
- stale broker orders are cancelled at the broker before being marked missed;
- multi-leg requests follow Alpaca's current MLEG contract: qty is the number of
  spread units, each leg carries its own side/position intent, and the parent
  request does not rely on a top-level side.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from execution.mleg_execution_manager import (
    MlegExecutionManager,
    MlegOrder,
    OrderState,
)

try:
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import TimeInForce, OrderClass, OrderSide, PositionIntent
    _ALPACA_OPTIONS_OK = True
except ImportError:  # pragma: no cover - exercised by environments without SDK
    _ALPACA_OPTIONS_OK = False


class V14_3_MlegExecutionManager(MlegExecutionManager):
    """Fail-closed MLEG execution manager for paper/live Alpaca communication."""

    def __init__(
        self,
        trading_client: Any = None,
        config: Optional[dict] = None,
        log_dir: str = "HFT/logs/v14_3",
        paper_mode: bool = True,
        allow_offline_simulation: bool = False,
    ):
        super().__init__(
            trading_client=trading_client,
            config=config,
            log_dir=log_dir,
            paper_mode=paper_mode,
        )
        self.allow_offline_simulation = bool(allow_offline_simulation)
        self.last_broker_error: str = ""

    @property
    def broker_backed(self) -> bool:
        return self.trading_client is not None

    def communication_probe(self) -> dict:
        """Read-only account/clock probe. Never submits or cancels an order."""
        result = {
            "ok": False,
            "account_ok": False,
            "clock_ok": False,
            "paper_mode": self.paper_mode,
            "error": "",
        }
        if self.trading_client is None:
            result["error"] = "no_trading_client"
            return result
        try:
            account = self.trading_client.get_account()
            result["account_ok"] = True
            result["account_status"] = str(getattr(account, "status", ""))
            result["options_approved_level"] = getattr(account, "options_approved_level", None)
            result["options_trading_level"] = getattr(account, "options_trading_level", None)
            result["options_buying_power"] = getattr(account, "options_buying_power", None)
            clock = self.trading_client.get_clock()
            result["clock_ok"] = True
            result["market_open"] = bool(getattr(clock, "is_open", False))
            result["ok"] = True
            self.last_broker_error = ""
        except Exception as exc:  # pragma: no cover - integration only
            self.last_broker_error = str(exc)
            result["error"] = self.last_broker_error
        return result

    def submit_order(self, order: MlegOrder) -> bool:
        """Submit entry; broker failure never becomes a synthetic fill."""
        if self.cfg.get("live_trading_enabled", False) is False and not self.paper_mode:
            order.state = OrderState.REJECTED
            order.cancel_reason = "live_trading_disabled_and_not_paper"
            self._log_order(order)
            return False

        order.submit_time = datetime.utcnow()
        order.state = OrderState.SUBMITTED
        order.last_updated = datetime.utcnow()

        if self.trading_client is not None:
            success = self._submit_to_broker(order)
            if not success:
                order.state = OrderState.REJECTED
                self.last_broker_error = order.cancel_reason
                self._release_contracts(order)
                self._log_order(order)
                return False
            # A fast paper fill can already be visible; otherwise leave SUBMITTED.
            self.check_order_status(order)
        elif self.paper_mode and self.allow_offline_simulation:
            self._simulate_fill(order)
        else:
            order.state = OrderState.REJECTED
            order.cancel_reason = "broker_required_offline_simulation_disabled"
            self._release_contracts(order)
            self._log_order(order)
            return False

        self._log_order(order)
        return True

    def submit_exit(self, order: MlegOrder) -> bool:
        """Submit close; broker failure never becomes a synthetic close."""
        order.submit_time = datetime.utcnow()
        order.state = OrderState.CLOSE_SUBMITTED
        order.last_updated = datetime.utcnow()

        if self.trading_client is not None:
            success = self._submit_to_broker(order)
            if not success:
                order.state = OrderState.CLOSE_FAILED
                self.last_broker_error = order.cancel_reason
                self._log_order(order)
                return False
            self.check_order_status(order)
        elif self.paper_mode and self.allow_offline_simulation:
            self._simulate_exit_fill(order)
        else:
            order.state = OrderState.CLOSE_FAILED
            order.cancel_reason = "broker_required_offline_simulation_disabled"
            self._log_order(order)
            return False

        self._log_order(order)
        return True

    def reconcile_positions(self) -> bool:
        """Use broker state whenever connected, including Alpaca paper accounts."""
        if self.trading_client is None:
            return bool(self.paper_mode and self.allow_offline_simulation)
        try:
            positions = self.trading_client.get_all_positions()
            self._broker_positions = {
                str(getattr(p, "symbol", "")): p for p in positions
            }
            self._last_reconcile_time = datetime.utcnow().timestamp()
            self.last_broker_error = ""
            return True
        except Exception as exc:  # pragma: no cover - integration only
            self.last_broker_error = str(exc)
            return False

    def cancel_stale_orders(self):
        """Cancel stale broker entries before declaring a missed fill."""
        now = datetime.utcnow()
        for order in list(self._orders.values()):
            if order.state not in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
                continue
            elapsed = (now - order.submit_time).total_seconds() if order.submit_time else 0.0
            if elapsed <= order.stale_timeout_seconds:
                continue

            if order.broker_order_id and self.trading_client is not None:
                try:
                    self.trading_client.cancel_order_by_id(order.broker_order_id)
                except Exception as exc:  # fail closed; do not pretend cancellation succeeded
                    self.last_broker_error = str(exc)
                    order.cancel_reason = f"stale_cancel_failed: {str(exc)[:160]}"
                    order.last_updated = now
                    self._log_order(order)
                    continue

            if order.state == OrderState.PARTIALLY_FILLED:
                # Partial fill means exposure can exist. Reconciliation is mandatory.
                order.cancel_reason = "partial_fill_stale_reconcile_required"
                order.last_updated = now
                self.reconcile_positions()
                self._log_order(order)
                continue

            order.state = OrderState.MISSED_FILL
            order.missed_fill_reason = f"stale_timeout_{elapsed:.0f}s"
            order.last_updated = now
            self._release_contracts(order)
            self._log_order(order)

    def _submit_to_broker(self, order: MlegOrder) -> bool:
        """Build the current Alpaca MLEG limit request and submit it."""
        if self.trading_client is None:
            order.cancel_reason = "no_trading_client"
            return False
        if not _ALPACA_OPTIONS_OK:
            order.cancel_reason = "alpaca_options_sdk_not_available"
            return False

        try:
            alpaca_legs = []
            for leg in order.legs:
                if leg.side == "buy_to_open":
                    side = OrderSide.BUY
                    intent = PositionIntent.BUY_TO_OPEN
                elif leg.side == "sell_to_open":
                    side = OrderSide.SELL
                    intent = PositionIntent.SELL_TO_OPEN
                elif leg.side == "buy_to_close":
                    side = OrderSide.BUY
                    intent = PositionIntent.BUY_TO_CLOSE
                elif leg.side == "sell_to_close":
                    side = OrderSide.SELL
                    intent = PositionIntent.SELL_TO_CLOSE
                else:
                    order.cancel_reason = f"unknown_leg_side: {leg.side}"
                    return False

                alpaca_legs.append(OptionLegRequest(
                    symbol=leg.contract_symbol,
                    ratio_qty=1.0,
                    side=side,
                    position_intent=intent,
                ))

            request = LimitOrderRequest(
                qty=float(order.quantity),
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                limit_price=round(float(order.intended_limit), 2),
                legs=alpaca_legs,
                client_order_id=order.client_order_id,
            )
            response = self.trading_client.submit_order(order_data=request)
            order.broker_order_id = str(getattr(response, "id", ""))
            if not order.broker_order_id:
                order.cancel_reason = "broker_returned_no_order_id"
                return False
            order.state = OrderState.SUBMITTED if order.direction == "OPEN" else OrderState.CLOSE_SUBMITTED
            order.submit_time = datetime.utcnow()
            self.last_broker_error = ""
            return True
        except Exception as exc:  # pragma: no cover - integration only
            order.cancel_reason = f"broker_error: {str(exc)[:200]}"
            self.last_broker_error = str(exc)
            return False
