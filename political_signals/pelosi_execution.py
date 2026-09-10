"""Broker execution for Pelosi disclosure-tail stock signals.

This module turns PelosiTailDecision objects into idempotent Alpaca stock orders.
It is deliberately separate from the disclosure parser/policy so source detection,
strategy logic, and broker writes remain auditable independently.

Execution modes:
- shadow: no broker writes
- paper: real orders sent to the Alpaca paper endpoint
- live: real-money Alpaca orders, enabled only by explicit environment gates

The executor trades regular hours only, never opens a naked short, never sells more
than strategy-owned quantity, reconciles fills from Alpaca, and persists a ledger so
restarts cannot duplicate a disclosure order.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    _ALPACA_OK = True
except Exception:  # pragma: no cover - unit tests inject a fake client
    TradingClient = None
    OrderSide = None
    TimeInForce = None
    MarketOrderRequest = None
    _ALPACA_OK = False

from alpaca_config import AlpacaConfig
from political_signals.pelosi_tail import PelosiTailDecision


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


@dataclass
class PelosiExecutionResult:
    disclosure_id: str
    ticker: str
    requested_action: str
    execution_action: str
    mode: str
    broker_order_id: str = ""
    client_order_id: str = ""
    requested_notional: float = 0.0
    requested_qty: float = 0.0
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    reason: str = ""
    timestamp: str = field(default_factory=lambda: _utc_now().isoformat())


class PelosiAlpacaExecutor:
    """Idempotent, strategy-owned stock execution for Pelosi tail signals."""

    VALID_MODES = {"shadow", "paper", "live"}
    TERMINAL = {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}

    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        trading_client: Any = None,
        state_path: str = "HFT/logs/pelosi_tail/execution_state.json",
        audit_path: str = "HFT/logs/pelosi_tail/execution_audit.jsonl",
        max_order_notional_pct: float = 0.08,
        max_symbol_equity_pct: float = 0.10,
        max_strategy_equity_pct: float = 0.20,
        min_order_notional: float = 5.0,
        max_pending_hours: float = 18.0,
    ):
        self.mode = str(mode or os.getenv("PELOSI_EXECUTION_MODE", "shadow")).strip().lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid PELOSI_EXECUTION_MODE={self.mode!r}")
        self._validate_mode_gate()

        self.max_order_notional_pct = float(max_order_notional_pct)
        self.max_symbol_equity_pct = float(max_symbol_equity_pct)
        self.max_strategy_equity_pct = float(max_strategy_equity_pct)
        self.min_order_notional = float(min_order_notional)
        self.max_pending_hours = float(max_pending_hours)

        self.state_path = Path(state_path)
        self.audit_path = Path(audit_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

        self.trading = trading_client
        if self.mode != "shadow" and self.trading is None:
            if not _ALPACA_OK:
                raise RuntimeError("alpaca-py trading client is unavailable")
            AlpacaConfig.validate()
            paper = self.mode == "paper"
            # paper= selects the correct Alpaca brokerage endpoint.  url_override is
            # intentionally avoided here so an accidentally stale BASE_URL cannot
            # redirect a live-mode decision to the wrong environment.
            self.trading = TradingClient(
                AlpacaConfig.API_KEY,
                AlpacaConfig.API_SECRET,
                paper=paper,
            )

    # ------------------------------------------------------------------
    # Configuration / persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _truthy(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}

    def _validate_mode_gate(self) -> None:
        if self.mode != "live":
            return
        if not self._truthy("PELOSI_ALLOW_LIVE"):
            raise RuntimeError(
                "Live Pelosi execution is locked. Set PELOSI_ALLOW_LIVE=true in addition "
                "to PELOSI_EXECUTION_MODE=live."
            )
        if bool(AlpacaConfig.PAPER):
            raise RuntimeError(
                "Live Pelosi execution requested while ALPACA_PAPER=true. Set ALPACA_PAPER=false "
                "only when intentionally using a live brokerage account."
            )

    def _load_state(self) -> Dict[str, Any]:
        default = {"version": 1, "events": {}, "orders": {}, "owned_lots": {}, "pending": {}}
        if not self.state_path.exists():
            return default
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            for key, value in default.items():
                data.setdefault(key, value.copy() if isinstance(value, dict) else value)
            return data
        except Exception:
            return default

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def _audit(self, payload: Dict[str, Any]) -> None:
        row = {"timestamp": _utc_now().isoformat(), "mode": self.mode, **payload}
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    # Broker/account helpers
    # ------------------------------------------------------------------
    def _clock_open(self) -> bool:
        if self.mode == "shadow":
            return True
        clock = self.trading.get_clock()
        return bool(getattr(clock, "is_open", False))

    def _account_snapshot(self) -> Dict[str, float]:
        account = self.trading.get_account()
        return {
            "equity": max(0.0, _f(getattr(account, "equity", 0.0))),
            "buying_power": max(0.0, _f(getattr(account, "buying_power", 0.0))),
        }

    def communication_probe(self) -> Dict[str, Any]:
        if self.mode == "shadow":
            return {"ok": True, "mode": "shadow", "broker_write_enabled": False}
        try:
            account = self.trading.get_account()
            clock = self.trading.get_clock()
            return {
                "ok": True,
                "mode": self.mode,
                "broker_write_enabled": True,
                "account_status": str(getattr(account, "status", "")),
                "equity": _f(getattr(account, "equity", 0.0)),
                "buying_power": _f(getattr(account, "buying_power", 0.0)),
                "market_open": bool(getattr(clock, "is_open", False)),
            }
        except Exception as exc:
            return {"ok": False, "mode": self.mode, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # Strategy-owned exposure
    # ------------------------------------------------------------------
    def owned_qty(self, symbol: str) -> float:
        symbol = symbol.upper()
        total = 0.0
        for lot in self.state["owned_lots"].get(symbol, []):
            remaining = max(0.0, _f(lot.get("remaining_qty", lot.get("filled_qty", 0.0))))
            total += remaining
        return total

    def _owned_cost(self, symbol: Optional[str] = None) -> float:
        symbols = [symbol.upper()] if symbol else list(self.state["owned_lots"].keys())
        total = 0.0
        for sym in symbols:
            for lot in self.state["owned_lots"].get(sym, []):
                qty = max(0.0, _f(lot.get("remaining_qty", 0.0)))
                px = max(0.0, _f(lot.get("filled_avg_price", 0.0)))
                total += qty * px
        return total

    def _record_fill_delta(self, order_id: str, broker_order: Any) -> None:
        rec = self.state["orders"].get(order_id)
        if not rec:
            return
        side = rec.get("side", "")
        symbol = rec.get("symbol", "").upper()
        new_filled = max(0.0, _f(getattr(broker_order, "filled_qty", 0.0)))
        prior_accounted = max(0.0, _f(rec.get("accounted_filled_qty", 0.0)))
        delta = max(0.0, new_filled - prior_accounted)
        avg = max(0.0, _f(getattr(broker_order, "filled_avg_price", 0.0)))
        status = _enum_value(getattr(broker_order, "status", ""))

        rec["broker_status"] = status
        rec["filled_qty"] = new_filled
        rec["filled_avg_price"] = avg
        rec["accounted_filled_qty"] = new_filled
        rec["last_reconciled_at"] = _utc_now().isoformat()

        if delta > 1e-9 and side == "BUY":
            self.state["owned_lots"].setdefault(symbol, []).append({
                "disclosure_id": rec.get("disclosure_id", ""),
                "order_id": order_id,
                "filled_qty": delta,
                "remaining_qty": delta,
                "filled_avg_price": avg,
                "filled_at": str(getattr(broker_order, "filled_at", "") or _utc_now().isoformat()),
            })

        if delta > 1e-9 and side == "SELL":
            to_remove = delta
            lots = self.state["owned_lots"].get(symbol, [])
            # FIFO only affects attribution; aggregate strategy quantity is unchanged.
            for lot in lots:
                if to_remove <= 1e-9:
                    break
                remaining = max(0.0, _f(lot.get("remaining_qty", 0.0)))
                cut = min(remaining, to_remove)
                lot["remaining_qty"] = max(0.0, remaining - cut)
                to_remove -= cut
            self.state["owned_lots"][symbol] = [
                x for x in lots if _f(x.get("remaining_qty", 0.0)) > 1e-9
            ]

    def reconcile(self) -> Dict[str, Any]:
        if self.mode == "shadow":
            return {"orders_checked": 0, "owned_qty": {}}
        checked = 0
        for order_id, rec in list(self.state["orders"].items()):
            status = str(rec.get("broker_status", "")).lower()
            # Filled orders are already fully accounted. Rejected/canceled zero-fill
            # orders need no further broker polling.
            if status == "filled" and _f(rec.get("filled_qty", 0.0)) <= _f(rec.get("accounted_filled_qty", 0.0)) + 1e-9:
                continue
            if status in self.TERMINAL and _f(rec.get("filled_qty", 0.0)) <= 0:
                continue
            try:
                broker_order = self.trading.get_order_by_id(order_id)
                self._record_fill_delta(order_id, broker_order)
                checked += 1
            except Exception as exc:
                rec["reconcile_error"] = f"{type(exc).__name__}: {exc}"
        self._expire_pending()
        self._save()
        return {
            "orders_checked": checked,
            "owned_qty": {sym: self.owned_qty(sym) for sym in self.state["owned_lots"]},
        }

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------
    def _remember_order(self, broker_order: Any, *, disclosure_id: str, symbol: str, side: str,
                        client_order_id: str, requested_notional: float = 0.0, requested_qty: float = 0.0) -> str:
        oid = str(getattr(broker_order, "id", "") or "")
        if not oid:
            raise RuntimeError("Alpaca returned an order without an id")
        self.state["orders"][oid] = {
            "disclosure_id": disclosure_id,
            "symbol": symbol.upper(),
            "side": side,
            "client_order_id": client_order_id,
            "requested_notional": float(requested_notional),
            "requested_qty": float(requested_qty),
            "broker_status": _enum_value(getattr(broker_order, "status", "new")),
            "filled_qty": max(0.0, _f(getattr(broker_order, "filled_qty", 0.0))),
            "filled_avg_price": max(0.0, _f(getattr(broker_order, "filled_avg_price", 0.0))),
            "accounted_filled_qty": 0.0,
            "submitted_at": _utc_now().isoformat(),
        }
        self._record_fill_delta(oid, broker_order)
        return oid

    def _submit_buy(self, d: PelosiTailDecision) -> PelosiExecutionResult:
        account = self._account_snapshot()
        equity = account["equity"]
        buying_power = account["buying_power"]
        if equity <= 0 or buying_power <= 0:
            return self._result(d, "BLOCKED", reason="no_buying_power")

        suggested = equity * max(0.0, float(d.target_notional_pct))
        max_single = equity * self.max_order_notional_pct
        max_symbol = max(0.0, equity * self.max_symbol_equity_pct - self._owned_cost(d.ticker))
        max_strategy = max(0.0, equity * self.max_strategy_equity_pct - self._owned_cost())
        notional = min(suggested, max_single, max_symbol, max_strategy, buying_power * 0.95)
        notional = math.floor(notional * 100.0) / 100.0
        if notional < self.min_order_notional:
            return self._result(d, "BLOCKED", reason="risk_budget_below_minimum")

        client_id = f"pelosi-{d.disclosure_id[:16]}-b"
        req = MarketOrderRequest(
            symbol=d.ticker,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_id,
        )
        broker = self.trading.submit_order(order_data=req)
        oid = self._remember_order(
            broker,
            disclosure_id=d.disclosure_id,
            symbol=d.ticker,
            side="BUY",
            client_order_id=client_id,
            requested_notional=notional,
        )
        self.state["events"][d.disclosure_id] = {
            "ticker": d.ticker,
            "action": "BUY",
            "order_id": oid,
            "submitted_at": _utc_now().isoformat(),
        }
        self.state["pending"].pop(d.disclosure_id, None)
        self._save()
        self._audit({"event": "BUY_SUBMITTED", "disclosure_id": d.disclosure_id, "ticker": d.ticker,
                     "order_id": oid, "notional": notional})
        return PelosiExecutionResult(
            disclosure_id=d.disclosure_id, ticker=d.ticker, requested_action=d.action,
            execution_action="BUY_SUBMITTED", mode=self.mode, broker_order_id=oid,
            client_order_id=client_id, requested_notional=notional,
        )

    def _cancel_pending_buys(self, symbol: str) -> None:
        symbol = symbol.upper()
        for oid, rec in self.state["orders"].items():
            if rec.get("symbol", "").upper() != symbol or rec.get("side") != "BUY":
                continue
            if str(rec.get("broker_status", "")).lower() in self.TERMINAL:
                continue
            try:
                self.trading.cancel_order_by_id(oid)
                rec["cancel_requested_at"] = _utc_now().isoformat()
            except Exception as exc:
                rec["cancel_error"] = f"{type(exc).__name__}: {exc}"
        # Reconcile any quantity that filled before cancellation.
        self.reconcile()

    def _submit_exit(self, d: PelosiTailDecision) -> PelosiExecutionResult:
        self._cancel_pending_buys(d.ticker)
        qty = self.owned_qty(d.ticker)
        if qty <= 1e-9:
            self.state["events"][d.disclosure_id] = {
                "ticker": d.ticker, "action": "EXIT_ONLY", "status": "NO_STRATEGY_POSITION",
                "processed_at": _utc_now().isoformat(),
            }
            self._save()
            return self._result(d, "NO_ACTION", reason="no_pelosi_owned_position")

        # Never sell more than the strategy believes it owns. Account-level checks may
        # include unrelated shares, so ownership is derived from our reconciled ledger.
        qty = math.floor(qty * 1_000_000.0) / 1_000_000.0
        client_id = f"pelosi-{d.disclosure_id[:16]}-s"
        req = MarketOrderRequest(
            symbol=d.ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_id,
        )
        broker = self.trading.submit_order(order_data=req)
        oid = self._remember_order(
            broker,
            disclosure_id=d.disclosure_id,
            symbol=d.ticker,
            side="SELL",
            client_order_id=client_id,
            requested_qty=qty,
        )
        self.state["events"][d.disclosure_id] = {
            "ticker": d.ticker,
            "action": "EXIT_ONLY",
            "order_id": oid,
            "submitted_at": _utc_now().isoformat(),
        }
        self._save()
        self._audit({"event": "EXIT_SUBMITTED", "disclosure_id": d.disclosure_id, "ticker": d.ticker,
                     "order_id": oid, "qty": qty})
        return PelosiExecutionResult(
            disclosure_id=d.disclosure_id, ticker=d.ticker, requested_action=d.action,
            execution_action="EXIT_SUBMITTED", mode=self.mode, broker_order_id=oid,
            client_order_id=client_id, requested_qty=qty,
        )

    # ------------------------------------------------------------------
    # Decision routing / queueing
    # ------------------------------------------------------------------
    def _result(self, d: PelosiTailDecision, action: str, *, reason: str = "") -> PelosiExecutionResult:
        return PelosiExecutionResult(
            disclosure_id=d.disclosure_id,
            ticker=d.ticker,
            requested_action=d.action,
            execution_action=action,
            mode=self.mode,
            reason=reason,
        )

    def _queue(self, d: PelosiTailDecision) -> PelosiExecutionResult:
        self.state["pending"][d.disclosure_id] = {
            "decision": asdict(d),
            "queued_at": _utc_now().isoformat(),
            "expires_at": (_utc_now() + timedelta(hours=self.max_pending_hours)).isoformat(),
        }
        self._save()
        self._audit({"event": "QUEUED_MARKET_CLOSED", "disclosure_id": d.disclosure_id, "ticker": d.ticker,
                     "action": d.action})
        return self._result(d, "QUEUED", reason="regular_market_closed")

    def _expire_pending(self) -> None:
        now = _utc_now()
        for did, item in list(self.state["pending"].items()):
            try:
                expires = datetime.fromisoformat(str(item.get("expires_at", "")).replace("Z", "+00:00"))
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if expires <= now:
                    self._audit({"event": "PENDING_EXPIRED", "disclosure_id": did})
                    self.state["pending"].pop(did, None)
            except Exception:
                self.state["pending"].pop(did, None)

    @staticmethod
    def _decision_from_dict(raw: Dict[str, Any]) -> PelosiTailDecision:
        return PelosiTailDecision(**raw)

    def process_pending(self) -> List[PelosiExecutionResult]:
        self._expire_pending()
        if self.mode == "shadow" or not self._clock_open():
            self._save()
            return []
        results = []
        for did, item in list(self.state["pending"].items()):
            if did in self.state["events"]:
                self.state["pending"].pop(did, None)
                continue
            try:
                decision = self._decision_from_dict(item["decision"])
                results.append(self.process(decision, allow_queue=False))
            except Exception as exc:
                self._audit({"event": "PENDING_EXECUTION_ERROR", "disclosure_id": did,
                             "error": f"{type(exc).__name__}: {exc}"})
        self._save()
        return results

    def process(self, d: PelosiTailDecision, *, allow_queue: bool = True) -> PelosiExecutionResult:
        # The disclosure fingerprint is the idempotency key. Never submit the same
        # public event twice, even across process restarts.
        if d.disclosure_id in self.state["events"]:
            return self._result(d, "DUPLICATE_IGNORED", reason="disclosure_already_processed")
        if d.action not in {"BUY", "EXIT_ONLY"}:
            return self._result(d, "NO_ACTION", reason=f"policy_action_{d.action.lower()}")
        if self.mode == "shadow":
            return self._result(d, "SHADOW_ONLY", reason="broker_writes_disabled")

        self.reconcile()
        if not self._clock_open():
            if allow_queue:
                return self._queue(d)
            return self._result(d, "WAITING_FOR_OPEN", reason="regular_market_closed")

        if d.action == "BUY":
            return self._submit_buy(d)
        return self._submit_exit(d)
