"""V14.5.1 broker execution hardening.

This manager extends the fail-closed V14.3 Alpaca MLEG path with explicit strategy
unit fill tracking and cancellation helpers used by V14.5.1's reconciliation
logic.  Multi-leg orders keep the spread atomic at the leg level, while parent
qty can still be partially filled across strategy units; those units must never
be lost from strategy state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from execution.mleg_execution_manager import MlegOrder, OrderState
from execution.v14_3_mleg_execution_manager import V14_3_MlegExecutionManager


class V14_5_1_MlegExecutionManager(V14_3_MlegExecutionManager):
    """Fail-closed Alpaca MLEG execution with parent-unit fill accounting."""

    @staticmethod
    def _status_text(obj: Any) -> str:
        value = getattr(obj, "value", obj)
        return str(value or "").lower().replace("orderstatus.", "")

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _capture_broker_fill_fields(self, order: MlegOrder, broker_order: Any) -> None:
        filled_qty = max(0.0, self._float(getattr(broker_order, "filled_qty", 0.0)))
        requested_qty = max(
            float(order.quantity),
            self._float(getattr(broker_order, "qty", order.quantity), float(order.quantity)),
        )
        fill_price = self._float(getattr(broker_order, "filled_avg_price", None), 0.0)
        previous = float(getattr(order, "broker_filled_qty", 0.0) or 0.0)
        order.broker_requested_qty = requested_qty
        order.broker_filled_qty = filled_qty
        order.broker_fill_delta_qty = max(0.0, filled_qty - previous)
        order.broker_remaining_qty = max(0.0, requested_qty - filled_qty)
        order.broker_filled_avg_price = abs(fill_price) if fill_price else 0.0
        order.last_broker_status = self._status_text(getattr(broker_order, "status", ""))

    def check_order_status(self, order: MlegOrder) -> OrderState:
        """Poll Alpaca and preserve filled_qty even when the parent is not complete."""
        if not self.trading_client or not order.broker_order_id:
            return order.state
        try:
            broker_order = self.trading_client.get_order_by_id(order.broker_order_id)
            self._capture_broker_fill_fields(order, broker_order)
            status = str(getattr(order, "last_broker_status", ""))
            filled_qty = float(getattr(order, "broker_filled_qty", 0.0) or 0.0)
            fill_price = float(getattr(order, "broker_filled_avg_price", 0.0) or 0.0)

            if status == "filled" or (order.quantity > 0 and filled_qty >= float(order.quantity)):
                order.state = OrderState.FILLED if order.direction == "OPEN" else OrderState.CLOSED
                order.fill_time = datetime.utcnow()
                if fill_price > 0:
                    if order.direction == "OPEN":
                        order.actual_fill_price = fill_price
                        order.fill_slippage_vs_mid = (
                            (fill_price - order.composite_mid) / order.composite_mid
                            if order.composite_mid > 0 else 0.0
                        )
                    else:
                        order.actual_exit_price = fill_price
                        order.exit_slippage_vs_bid = (
                            (order.composite_bid - fill_price) / order.composite_bid
                            if order.composite_bid > 0 else 0.0
                        )
                self._log_fill(order)
            elif status == "partially_filled" or (0 < filled_qty < float(order.quantity)):
                order.state = OrderState.PARTIALLY_FILLED
                if fill_price > 0:
                    if order.direction == "OPEN":
                        order.actual_fill_price = fill_price
                    else:
                        order.actual_exit_price = fill_price
            elif status in ("canceled", "cancelled"):
                # A canceled parent can still have strategy units filled.  The
                # caller inspects broker_filled_qty before deciding whether this
                # is a missed fill or a live position.
                order.state = OrderState.CANCELED
                if not order.cancel_reason:
                    order.cancel_reason = "broker_canceled"
            elif status == "expired":
                order.state = OrderState.EXPIRED
            elif status in ("rejected", "suspended"):
                order.state = OrderState.REJECTED if order.direction == "OPEN" else OrderState.CLOSE_FAILED
                if not order.cancel_reason:
                    order.cancel_reason = f"broker_{status}"
            elif status in ("new", "accepted", "pending_new", "accepted_for_bidding", "pending_replace"):
                order.state = OrderState.SUBMITTED if order.direction == "OPEN" else OrderState.CLOSE_SUBMITTED

            order.last_updated = datetime.utcnow()
            self.last_broker_error = ""
        except Exception as exc:  # pragma: no cover - broker integration only
            self.last_broker_error = str(exc)
        return order.state

    def cancel_remainder(self, order: MlegOrder, reason: str) -> bool:
        """Cancel only the unfilled remainder; never pretend filled units vanished."""
        if not self.trading_client or not order.broker_order_id:
            return False
        try:
            self.trading_client.cancel_order_by_id(order.broker_order_id)
            order.cancel_reason = str(reason)
            order.last_updated = datetime.utcnow()
            return True
        except Exception as exc:  # pragma: no cover
            self.last_broker_error = str(exc)
            order.cancel_reason = f"cancel_remainder_failed:{str(exc)[:160]}"
            order.last_updated = datetime.utcnow()
            self._log_order(order)
            return False

    def strategy_units_at_broker(self, order: MlegOrder) -> int:
        """Best-effort leg reconciliation for a 1:1 vertical.

        V14.5 forbids pyramiding a symbol, so the minimum absolute position across
        the two option legs is a conservative estimate of complete spread units.
        """
        if self.trading_client is None or len(order.legs) < 2:
            return int(float(getattr(order, "broker_filled_qty", 0.0) or 0.0))
        try:
            positions = self.trading_client.get_all_positions()
            by_symbol = {str(getattr(p, "symbol", "")): p for p in positions}
            quantities = []
            for leg in order.legs[:2]:
                pos = by_symbol.get(leg.contract_symbol)
                if pos is None:
                    quantities.append(0)
                    continue
                quantities.append(int(abs(self._float(getattr(pos, "qty", 0.0)))))
            self._broker_positions = by_symbol
            self._last_reconcile_time = datetime.utcnow().timestamp()
            return min(quantities) if quantities else 0
        except Exception as exc:  # pragma: no cover
            self.last_broker_error = str(exc)
            return int(float(getattr(order, "broker_filled_qty", 0.0) or 0.0))

    def mark_emergency_exit_required(self, order: MlegOrder, reason: str) -> None:
        order.state = OrderState.EMERGENCY_EXIT_REQUESTED
        order.cancel_reason = str(reason)
        order.last_updated = datetime.utcnow()
        self._log_order(order)
