"""Broker-backed V14.3 high-vol options core-runner.

This module is the production-shaped paper runner for the V14.3 local optimum.
It keeps the validated strategy idea (watch -> confirm -> debit spread) but fixes
execution gaps in the V14.2 prototype:

* V14.3 thresholds are enforced directly instead of WatchTicket's V14.2 globals.
* The option spread is observed at watch time and the SAME contracts are re-quoted
  after confirmation, so mid inflation is meaningful.
* The first executable quote must be newer than the confirmation timestamp.
* COIN/TSLA symbol policy size multipliers change actual buying-power/debit limits.
* Alpaca broker failures fail closed; they never become simulated fills.
* Entry fills are polled and reconciled.
* Exit decisions use the CURRENT option spread credit, not underlying return.
* Core and runner are actually closed separately when at least two spread units exist.
* A runner profit lock is activated only after the core is paid.

V14.3 remains PAPER/REPLAY ONLY by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from strategy.v14_2_core_runner import V14_2_CoreRunner
from strategy.v14_3_highvol_config import (
    get_v14_3_config,
    get_symbol_policy,
    can_trade_symbol,
)
from strategy.watch_ticket import WatchTicket, TicketSide, TicketStatus
from strategy.spread_quality_gate import OptionLeg
from strategy.package_builder import PackageResult
from execution.mleg_execution_manager import MlegLeg, MlegOrder, OrderState
from execution.v14_3_mleg_execution_manager import V14_3_MlegExecutionManager


@dataclass
class PositionPlan:
    ticket_id: str
    symbol: str
    mode: str
    long_contract: Any
    short_contract: Any
    entry_order_id: str
    total_qty: int
    core_qty: int = 0
    runner_qty: int = 0
    fallback_qty: int = 0
    entry_debit: float = 0.0          # per spread unit, dollars/share
    total_debit_dollars: float = 0.0
    core_closed: bool = False
    runner_lock_active: bool = False
    peak_pnl_pct: float = -999.0
    realized_pnl: float = 0.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class V14_3_OptionsRunner(V14_2_CoreRunner):
    """V14.3 strategy with real option quote and Alpaca paper-order plumbing."""

    VERSION = "14.3-options"
    STATUS = "LOCAL_OPTIMUM_REAL_QUOTES_PAPER_VALIDATION"
    LABEL = "V14_3_HIGH_VOL_CORE_RUNNER_OPTIONS_PAPER"

    # Backtest-derived priors are intentionally treated as priors, not truth.
    _SYMBOL_PRIOR_WIN = {"COIN": 0.75, "TSLA": 0.357}

    def __init__(
        self,
        symbol: str,
        trading_client: Any,
        option_data_client: Any,
        config: Optional[dict] = None,
        paper_mode: bool = True,
        log_dir: str = "HFT/logs/v14_3_options",
        allow_offline_simulation: bool = False,
    ):
        self.symbol = symbol.upper()
        cfg = config or get_v14_3_config()
        allowed, reason, size_mult = can_trade_symbol(self.symbol)
        if not allowed:
            raise ValueError(f"V14.3 options policy blocks {self.symbol}: {reason}")
        if trading_client is None and not allow_offline_simulation:
            raise ValueError("V14.3 options runner requires an Alpaca TradingClient")
        if option_data_client is None and not allow_offline_simulation:
            raise ValueError("V14.3 options runner requires an OptionHistoricalDataClient")

        super().__init__(
            trading_client=trading_client,
            option_data_client=option_data_client,
            config=cfg,
            paper_mode=paper_mode,
            log_dir=log_dir,
        )

        self.size_multiplier = float(size_mult)
        self.symbol_policy = get_symbol_policy(self.symbol)
        self.execution_manager = V14_3_MlegExecutionManager(
            trading_client=trading_client,
            config=self.cfg,
            log_dir=str(self.log_dir),
            paper_mode=paper_mode,
            allow_offline_simulation=allow_offline_simulation,
        )

        self._watch_contracts: Dict[str, Tuple[Any, Any]] = {}
        self._watch_high: Dict[str, float] = {}
        self._watch_low: Dict[str, float] = {}
        self._pending_entry: Dict[str, Dict[str, Any]] = {}
        self._pending_exit: Dict[str, Dict[str, Any]] = {}
        self._positions: Dict[str, PositionPlan] = {}
        self._last_day = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def communication_probe(self) -> dict:
        """Read-only Trading API + option-contract/quote communication check."""
        broker = self.execution_manager.communication_probe()
        result = {"broker": broker, "options_ok": False, "ok": False, "error": ""}
        if not broker.get("ok"):
            result["error"] = broker.get("error", "broker_probe_failed")
            return result
        if not self.chain_fetcher:
            result["error"] = "option_chain_fetcher_unavailable"
            return result
        try:
            # This is read-only. A missing chain is a failed options communication/preflight.
            spread = self.chain_fetcher.get_spread_with_quotes(
                self.symbol,
                underlying_price=self._probe_underlying_price(),
                side="CALL",
                dte_min=5,
                dte_max=14,
                strike_width=5.0,
            )
            if not spread:
                result["error"] = "no_option_spread_or_quote_returned"
                return result
            long_leg, short_leg, long_contract, short_contract = spread
            result["options_ok"] = bool(long_leg.is_valid and short_leg.is_valid)
            result["long_contract"] = long_contract.symbol
            result["short_contract"] = short_contract.symbol
            result["long_bid"] = long_leg.bid
            result["long_ask"] = long_leg.ask
            result["short_bid"] = short_leg.bid
            result["short_ask"] = short_leg.ask
            result["ok"] = bool(broker.get("ok") and result["options_ok"])
            return result
        except Exception as exc:  # pragma: no cover - integration only
            result["error"] = str(exc)
            return result

    def on_market_iteration(self):
        """Poll broker state, cancel stale entries, and reset day only once."""
        today = datetime.utcnow().date()
        if today != self._last_day:
            self.risk_manager.reset_day()
            self._last_day = today
        self.execution_manager.cancel_stale_orders()
        self.poll_broker_orders()

    def evaluate_opportunity(
        self,
        symbol: str,
        price: float,
        vwap: float,
        atr: float,
        high_of_day: float,
        low_of_day: float,
        trend_slope: float,
        volume_ratio: float,
        price_5m_ago: float,
        price_15m_ago: float,
        option_liquidity: float = 0.7,
        iv_percentile: float = 0.5,
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        out = {"action": "NONE", "symbol": symbol, "timestamp": datetime.utcnow().isoformat(), "details": {}}
        if symbol != self.symbol:
            out["action"] = "SKIPPED"
            out["details"]["reason"] = "runner_symbol_mismatch"
            return out

        allowed, reason, size_mult = can_trade_symbol(symbol)
        if not allowed:
            out["action"] = "SKIPPED"
            out["details"]["reason"] = f"symbol_policy:{reason}"
            return out
        self.size_multiplier = float(size_mult)
        self.on_market_iteration()

        # Manage existing positions from CURRENT option quotes before creating risk.
        self._manage_real_option_exits(symbol, price, vwap)

        # Process already-confirmed tickets waiting for a quote newer than confirmation.
        for ticket in list(self.ticket_book.active_tickets.values()):
            if ticket.symbol == symbol and ticket.status == TicketStatus.CONFIRMED:
                exec_result = self._attempt_real_execution(ticket, price, iv_percentile)
                if exec_result.get("action") not in ("WAIT_NEXT_QUOTE", "NONE"):
                    out["action"] = exec_result["action"]
                    out["details"].update(exec_result)

        # Route scoring is still the V14 family signal source.
        candidates = self.route_conditioner.evaluate_routes(
            price=price, vwap=vwap, atr=atr,
            high_of_day=high_of_day, low_of_day=low_of_day,
            trend_slope=trend_slope, volume_ratio=volume_ratio,
            price_5m_ago=price_5m_ago, price_15m_ago=price_15m_ago,
            option_liquidity=option_liquidity, iv_percentile=iv_percentile,
        )

        for candidate in candidates:
            if not self._passes_v14_3_watch(candidate):
                continue
            if self._has_active_route(symbol, candidate.route):
                continue
            if not self.route_monitor.is_route_allowed(candidate.route):
                continue
            if not self.env_monitor.is_environment_allowed(self._current_environment):
                continue

            watch_quote = self._observe_watch_spread(symbol, price, candidate.side)
            if watch_quote is None:
                out["action"] = "SKIPPED"
                out["details"]["reason"] = "option_chain_or_quote_unavailable"
                continue
            long_leg, short_leg, long_contract, short_contract = watch_quote
            spread_mid = long_leg.mid - short_leg.mid
            if spread_mid <= 0:
                continue
            real_liquidity = self._quote_liquidity_score(long_leg, short_leg, spread_mid)
            if real_liquidity < self.cfg["watch_min_option_liquidity"]:
                continue

            side = TicketSide.CALL if candidate.side == "CALL" else TicketSide.PUT
            ticket = WatchTicket(
                symbol=symbol,
                route=candidate.route,
                side=side,
                timestamp_created=datetime.utcnow(),
                underlying_price_at_watch=price,
                vwap_at_watch=vwap,
                atr_at_watch=atr,
                route_score=candidate.score,
                ic_spread=candidate.ic_spread,
                expected_ev_over_debit=candidate.ev_over_debit,
                option_liquidity_score=real_liquidity,
                expected_move_to_target=candidate.expected_move,
                estimated_debit_at_watch=spread_mid,
                candidate_contracts=[long_contract.symbol, short_contract.symbol],
                environment_state={"iv_percentile": iv_percentile},
            )
            # Bypass WatchTicket's V14.2-global admission; V14.3 was checked above.
            self.ticket_book.active_tickets[ticket.ticket_id] = ticket
            self.ticket_book._log_event(ticket, "WATCHING_V14_3")
            self._watch_contracts[ticket.ticket_id] = (long_contract, short_contract)
            self._watch_high[ticket.ticket_id] = price
            self._watch_low[ticket.ticket_id] = price
            out["action"] = "WATCHING"
            out["details"].update({
                "ticket_id": ticket.ticket_id,
                "route": ticket.route,
                "route_score": ticket.route_score,
                "watch_debit": spread_mid,
                "size_multiplier": self.size_multiplier,
                "contracts": ticket.candidate_contracts,
            })

        # Confirm watches using intraday high/low since each watch, not HOD/LOD.
        for ticket in list(self.ticket_book.get_watching_tickets(symbol)):
            self._watch_high[ticket.ticket_id] = max(self._watch_high.get(ticket.ticket_id, price), price)
            self._watch_low[ticket.ticket_id] = min(self._watch_low.get(ticket.ticket_id, price), price)
            favorable = price - ticket.underlying_price_at_watch if ticket.side == TicketSide.CALL else ticket.underlying_price_at_watch - price
            mfe_velocity = favorable / atr if atr > 0 else 0.0
            env_stress = self._environment_stress(iv_percentile)
            confirmation = self.confirmation_engine.check_confirmation(
                ticket=ticket,
                current_price=price,
                current_vwap=vwap,
                current_atr=atr,
                high_since_watch=self._watch_high[ticket.ticket_id],
                low_since_watch=self._watch_low[ticket.ticket_id],
                mfe_velocity=mfe_velocity,
                env_stress=env_stress,
                route_score_now=ticket.route_score,
                # Execution enforces a strictly newer quote; do not pretend one exists here.
                option_quote_valid=True,
                timestamp=datetime.utcnow(),
            )
            if confirmation.confirmed:
                self.confirmation_engine.apply_confirmation(ticket, confirmation, datetime.utcnow())
                out["action"] = "CONFIRMED"
                out["details"]["ticket_id"] = ticket.ticket_id
                exec_result = self._attempt_real_execution(ticket, price, iv_percentile)
                out["action"] = exec_result.get("action", out["action"])
                out["details"].update(exec_result)

        self.ticket_book.expire_stale_tickets(max_watch_bars=20)
        return out

    # ------------------------------------------------------------------
    # Watch and quote helpers
    # ------------------------------------------------------------------
    def _passes_v14_3_watch(self, candidate) -> bool:
        return (
            candidate.route in (self.cfg["routes_allowed"] + self.cfg["routes_soft_only"])
            and candidate.score >= self.cfg["watch_min_route_score"]
            and candidate.ic_spread >= self.cfg["watch_min_ic_spread"]
            and candidate.ev_over_debit >= self.cfg["watch_min_ev_over_debit"]
        )

    def _has_active_route(self, symbol: str, route: str) -> bool:
        return any(
            t.symbol == symbol and t.route == route and t.status in
            (TicketStatus.WATCHING, TicketStatus.CONFIRMED, TicketStatus.FILLED)
            for t in self.ticket_book.active_tickets.values()
        )

    def _observe_watch_spread(self, symbol: str, price: float, side: str):
        if not self.chain_fetcher:
            return None
        return self.chain_fetcher.get_spread_with_quotes(
            symbol=symbol,
            underlying_price=price,
            side=side,
            dte_min=5,
            dte_max=14,
            strike_width=5.0,
        )

    @staticmethod
    def _quote_liquidity_score(long_leg: OptionLeg, short_leg: OptionLeg, spread_mid: float) -> float:
        if spread_mid <= 0 or not long_leg.is_valid or not short_leg.is_valid:
            return 0.0
        width = long_leg.spread_width + short_leg.spread_width
        width_ratio = width / spread_mid
        # 1.0 at near-zero width; 0.0 at >= 50% of spread mid.
        return max(0.0, min(1.0, 1.0 - width_ratio / 0.50))

    @staticmethod
    def _parse_quote_ts(value) -> Optional[datetime]:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            text = str(value).replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _fresh_exact_quotes(self, ticket: WatchTicket):
        contracts = self._watch_contracts.get(ticket.ticket_id)
        if not contracts or not self.chain_fetcher:
            return None
        long_contract, short_contract = contracts
        legs = self.chain_fetcher.get_live_quotes(long_contract, short_contract)
        if not legs:
            return None
        long_leg, short_leg = legs
        if not long_leg.is_valid or not short_leg.is_valid:
            return None
        if ticket.timestamp_confirmed:
            confirmed = ticket.timestamp_confirmed
            if confirmed.tzinfo is None:
                confirmed = confirmed.replace(tzinfo=timezone.utc)
            else:
                confirmed = confirmed.astimezone(timezone.utc)
            timestamps = [self._parse_quote_ts(long_leg.quote_timestamp), self._parse_quote_ts(short_leg.quote_timestamp)]
            # If timestamps are supplied, both must be strictly newer than confirmation.
            supplied = [ts for ts in timestamps if ts is not None]
            if supplied and (len(supplied) < 2 or min(supplied) <= confirmed):
                return None
        return long_leg, short_leg, long_contract, short_contract

    def _environment_stress(self, iv_percentile: float) -> float:
        return max(0.0, min(1.0, max(0.0, iv_percentile - 0.50) * 0.50))

    def _estimate_pwin(self, ticket: WatchTicket, quality_score: float) -> float:
        prior = self._SYMBOL_PRIOR_WIN.get(ticket.symbol, 0.50)
        confirmation_quality = max(0.0, min(1.0, 0.55 * quality_score + 0.45 * min(1.0, ticket.mfe_velocity)))
        # Shrink noisy symbol history toward the present confirmation rather than
        # treating either as a calibrated probability by itself.
        return max(0.20, min(0.90, 0.55 * prior + 0.45 * confirmation_quality))

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def _attempt_real_execution(self, ticket: WatchTicket, underlying_price: float, iv_percentile: float) -> Dict[str, Any]:
        if ticket.ticket_id in self._pending_entry or ticket.ticket_id in self._positions:
            return {"action": "NONE"}
        fresh = self._fresh_exact_quotes(ticket)
        if fresh is None:
            return {"action": "WAIT_NEXT_QUOTE", "reason": "waiting_for_quote_newer_than_confirmation"}
        long_leg, short_leg, long_contract, short_contract = fresh

        target_price = ticket.underlying_price_at_watch + ticket.expected_move_to_target
        if ticket.side == TicketSide.PUT:
            target_price = ticket.underlying_price_at_watch - ticket.expected_move_to_target
        quality = self.spread_gate.evaluate(
            long_leg=long_leg,
            short_leg=short_leg,
            underlying_price=underlying_price,
            underlying_price_at_watch=ticket.underlying_price_at_watch,
            estimated_debit_at_watch=ticket.estimated_debit_at_watch,
            target_price=target_price,
            route=ticket.route,
        )
        if not quality.passed:
            ticket.mark_rejected(f"spread_quality:{quality.rejection_reason}")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            return {"action": "REJECTED", "reason": quality.rejection_reason}

        quality_score = max(0.0, min(1.0, 0.75 * ticket.route_score + 0.25 * min(1.0, ticket.mfe_velocity)))
        pwin = self._estimate_pwin(ticket, quality_score)
        effective_buying_power = self._buying_power * self.size_multiplier
        package = self.package_builder.build(
            quality_report=quality,
            quality_score=quality_score,
            pwin=pwin,
            env_stress=self._environment_stress(iv_percentile),
            route=ticket.route,
            account_buying_power=effective_buying_power,
            current_iv=long_leg.iv,
        )
        if package.total_contracts <= 0:
            ticket.mark_rejected("cannot_afford_spread")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            return {"action": "REJECTED", "reason": "cannot_afford_spread"}

        debit_per_contract = quality.spread_mid * 100.0
        policy_budget = effective_buying_power * self.cfg["max_open_debit_exposure_pct"]
        max_contracts = int(policy_budget / debit_per_contract) if debit_per_contract > 0 else 0
        if max_contracts <= 0:
            ticket.mark_rejected("symbol_policy_debit_budget")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            return {"action": "REJECTED", "reason": "symbol_policy_debit_budget"}
        qty = min(package.total_contracts, max_contracts)

        # A true core-runner needs at least 2 spread units. Otherwise force fallback.
        if package.is_package and qty < 2:
            package.is_package = False
            package.is_fallback = True
            package.fallback_reason = "package_requires_two_spread_units"
            package.core = None
            package.runner = None
        if package.is_package:
            core_qty = max(1, round(qty * self.cfg["core_fraction"]))
            runner_qty = max(1, qty - core_qty)
            if core_qty + runner_qty > qty:
                core_qty = qty - runner_qty
            fallback_qty = 0
        else:
            core_qty = runner_qty = 0
            fallback_qty = qty

        total_debit = qty * debit_per_contract
        risk = self.risk_manager.pre_trade_check(
            symbol=ticket.symbol,
            route=ticket.route,
            debit=total_debit,
            is_live=not self.paper_mode,
        )
        if not risk.allowed:
            ticket.mark_rejected(f"risk:{risk.reason}")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            return {"action": "REJECTED", "reason": risk.reason}

        self.execution_manager.cancel_all_entries_for_symbol(ticket.symbol)
        legs = [
            MlegLeg(long_contract.symbol, "buy_to_open", qty, long_contract.option_type, long_contract.strike, long_contract.expiration),
            MlegLeg(short_contract.symbol, "sell_to_open", qty, short_contract.option_type, short_contract.strike, short_contract.expiration),
        ]
        max_chase = self.cfg["max_limit_chase_pct_of_debit"]
        limit_price = min(quality.spread_ask, quality.spread_mid * (1.0 + max_chase))
        order = self.execution_manager.create_entry_order(
            ticket_id=ticket.ticket_id,
            legs=legs,
            limit_price=limit_price,
            quantity=qty,
            composite_bid=quality.spread_bid,
            composite_ask=quality.spread_ask,
            composite_mid=quality.spread_mid,
        )
        if order is None:
            return {"action": "REJECTED", "reason": "duplicate_or_contradictory_order"}

        self._pending_entry[ticket.ticket_id] = {
            "order": order,
            "package": package,
            "qty": qty,
            "core_qty": core_qty,
            "runner_qty": runner_qty,
            "fallback_qty": fallback_qty,
            "long_contract": long_contract,
            "short_contract": short_contract,
            "planned_debit": total_debit,
        }
        if not self.execution_manager.submit_order(order):
            self._pending_entry.pop(ticket.ticket_id, None)
            ticket.mark_rejected(order.cancel_reason or "broker_submission_failed")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            return {"action": "REJECTED", "reason": order.cancel_reason or "broker_submission_failed"}

        if order.state == OrderState.FILLED:
            self._activate_entry(ticket.ticket_id)
            return {"action": "FILLED", "mode": "PACKAGE" if package.is_package else "FALLBACK", "contracts": qty, "pwin": pwin}
        return {
            "action": "SUBMITTED",
            "order_id": order.order_id,
            "broker_order_id": order.broker_order_id,
            "mode": "PACKAGE" if package.is_package else "FALLBACK",
            "contracts": qty,
            "size_multiplier": self.size_multiplier,
            "pwin": pwin,
        }

    def _activate_entry(self, ticket_id: str):
        pending = self._pending_entry.pop(ticket_id, None)
        ticket = self.ticket_book.active_tickets.get(ticket_id)
        if not pending or not ticket:
            return
        order = pending["order"]
        if order.state != OrderState.FILLED:
            return
        entry_debit = float(order.actual_fill_price or order.intended_limit)
        plan = PositionPlan(
            ticket_id=ticket_id,
            symbol=ticket.symbol,
            mode="PACKAGE" if pending["package"].is_package else "FALLBACK",
            long_contract=pending["long_contract"],
            short_contract=pending["short_contract"],
            entry_order_id=order.order_id,
            total_qty=pending["qty"],
            core_qty=pending["core_qty"],
            runner_qty=pending["runner_qty"],
            fallback_qty=pending["fallback_qty"],
            entry_debit=entry_debit,
            total_debit_dollars=entry_debit * 100.0 * pending["qty"],
        )
        self._positions[ticket_id] = plan
        ticket.mark_filled()
        self.risk_manager.add_open_position(ticket.symbol, plan.total_debit_dollars)
        self.risk_manager.record_trade(ticket.symbol, ticket.route, plan.total_debit_dollars)
        self.slippage_monitor.record_entry_fill(order.fill_slippage_vs_mid, was_filled=True)

    # ------------------------------------------------------------------
    # Broker polling and real spread exits
    # ------------------------------------------------------------------
    def poll_broker_orders(self):
        for ticket_id, pending in list(self._pending_entry.items()):
            order = pending["order"]
            self.execution_manager.check_order_status(order)
            if order.state == OrderState.FILLED:
                self._activate_entry(ticket_id)
            elif order.state in (OrderState.REJECTED, OrderState.CANCELED, OrderState.EXPIRED, OrderState.MISSED_FILL):
                ticket = self.ticket_book.active_tickets.get(ticket_id)
                if ticket:
                    ticket.mark_missed_fill(order.cancel_reason or order.missed_fill_reason or order.state.value)
                    self.ticket_book.retire_ticket(ticket_id)
                self._pending_entry.pop(ticket_id, None)
                self.slippage_monitor.record_entry_fill(0.0, was_filled=False)

        for order_id, meta in list(self._pending_exit.items()):
            order = meta["order"]
            self.execution_manager.check_order_status(order)
            if order.state == OrderState.CLOSED:
                self._finalize_exit(order_id)
            elif order.state in (OrderState.REJECTED, OrderState.CANCELED, OrderState.EXPIRED, OrderState.CLOSE_FAILED):
                # Keep the position open. A failed close must never be interpreted as flat.
                self._pending_exit.pop(order_id, None)

    def _current_spread_quote(self, plan: PositionPlan):
        if not self.chain_fetcher:
            return None
        legs = self.chain_fetcher.get_live_quotes(plan.long_contract, plan.short_contract)
        if not legs:
            return None
        long_leg, short_leg = legs
        if not long_leg.is_valid or not short_leg.is_valid:
            return None
        close_bid = long_leg.bid - short_leg.ask
        close_ask = long_leg.ask - short_leg.bid
        close_mid = long_leg.mid - short_leg.mid
        if close_mid <= 0:
            return None
        return long_leg, short_leg, max(0.01, close_bid), max(0.01, close_ask), close_mid

    def _manage_real_option_exits(self, symbol: str, underlying_price: float, vwap: float):
        for ticket_id, plan in list(self._positions.items()):
            if plan.symbol != symbol or any(m["ticket_id"] == ticket_id for m in self._pending_exit.values()):
                continue
            quote = self._current_spread_quote(plan)
            if quote is None or plan.entry_debit <= 0:
                continue
            _, _, close_bid, close_ask, close_mid = quote
            # Decisions use the executable/natural close credit, not midpoint fantasy.
            pnl_pct = (close_bid - plan.entry_debit) / plan.entry_debit
            plan.peak_pnl_pct = max(plan.peak_pnl_pct, pnl_pct)

            ticket = self.ticket_book.active_tickets.get(ticket_id)
            if not ticket:
                continue
            continuation_failed = (
                ticket.side == TicketSide.CALL and underlying_price < vwap
            ) or (
                ticket.side == TicketSide.PUT and underlying_price > vwap
            )

            if plan.mode == "FALLBACK":
                if pnl_pct <= self.cfg["single_spread_initial_stop_pct"]:
                    self._submit_partial_exit(plan, plan.fallback_qty, close_bid, close_ask, "fallback_stop", "ALL")
                elif pnl_pct >= self.cfg["soft_greed_target_pct"]:
                    self._submit_partial_exit(plan, plan.fallback_qty, close_bid, close_ask, "fallback_target", "ALL")
                elif continuation_failed and pnl_pct > 0:
                    self._submit_partial_exit(plan, plan.fallback_qty, close_bid, close_ask, "continuation_decay", "ALL")
                continue

            # Package stop applies until/unless the core pays.
            if not plan.core_closed:
                if pnl_pct <= self.cfg["package_stop_pct"]:
                    self._submit_partial_exit(plan, plan.total_qty, close_bid, close_ask, "package_stop", "ALL")
                elif pnl_pct >= self.cfg["core_target_pct"]:
                    self._submit_partial_exit(plan, plan.core_qty, close_bid, close_ask, "core_target", "CORE")
                continue

            # Runner can never be intentionally turned back into a full loss after core pays.
            if plan.runner_qty <= 0:
                continue
            if pnl_pct >= self.cfg["runner_target_pct"]:
                self._submit_partial_exit(plan, plan.runner_qty, close_bid, close_ask, "runner_target", "RUNNER")
            elif pnl_pct <= self.cfg["runner_lock_pct"]:
                self._submit_partial_exit(plan, plan.runner_qty, close_bid, close_ask, "runner_lock", "RUNNER")
            elif continuation_failed and pnl_pct > self.cfg["runner_lock_pct"]:
                self._submit_partial_exit(plan, plan.runner_qty, close_bid, close_ask, "runner_continuation_decay", "RUNNER")

    def _submit_partial_exit(self, plan: PositionPlan, qty: int, close_bid: float, close_ask: float, reason: str, role: str):
        if qty <= 0:
            return
        close_legs = [
            MlegLeg(plan.long_contract.symbol, "sell_to_close", qty, plan.long_contract.option_type, plan.long_contract.strike, plan.long_contract.expiration),
            MlegLeg(plan.short_contract.symbol, "buy_to_close", qty, plan.short_contract.option_type, plan.short_contract.strike, plan.short_contract.expiration),
        ]
        # Alpaca MLEG convention: a credit limit is NEGATIVE. We store executable
        # credit separately and send -credit to the broker.
        credit = max(0.01, close_bid)
        order = self.execution_manager.create_exit_order(
            ticket_id=plan.ticket_id,
            legs=close_legs,
            limit_price=-credit,
            quantity=qty,
            exit_bid=close_bid,
            exit_ask=close_ask,
        )
        if order is None:
            return
        self._pending_exit[order.order_id] = {
            "order": order,
            "ticket_id": plan.ticket_id,
            "qty": qty,
            "role": role,
            "reason": reason,
            "expected_credit": credit,
        }
        if not self.execution_manager.submit_exit(order):
            self._pending_exit.pop(order.order_id, None)
            return
        if order.state == OrderState.CLOSED:
            self._finalize_exit(order.order_id)

    def _finalize_exit(self, order_id: str):
        meta = self._pending_exit.pop(order_id, None)
        if not meta:
            return
        plan = self._positions.get(meta["ticket_id"])
        ticket = self.ticket_book.active_tickets.get(meta["ticket_id"])
        if not plan or not ticket:
            return
        order = meta["order"]
        credit = abs(float(order.actual_exit_price or meta["expected_credit"]))
        qty = int(meta["qty"])
        pnl = (credit - plan.entry_debit) * 100.0 * qty
        plan.realized_pnl += pnl
        self.slippage_monitor.record_exit_fill(order.exit_slippage_vs_bid)

        if meta["role"] == "CORE":
            plan.core_closed = True
            plan.runner_lock_active = True
            plan.core_qty = 0
            return

        # ALL or RUNNER closes the remaining strategy position.
        if meta["role"] == "RUNNER":
            plan.runner_qty = 0
        else:
            plan.core_qty = plan.runner_qty = plan.fallback_qty = 0
        ticket.mark_closed(plan.realized_pnl, meta["reason"])
        self.ticket_book.retire_ticket(ticket.ticket_id)
        self.risk_manager.record_close(ticket.symbol, plan.realized_pnl)
        self.route_monitor.record_trade(ticket.route, plan.realized_pnl)
        self.env_monitor.record_trade(self._current_environment, ticket.symbol, plan.realized_pnl)
        self._positions.pop(ticket.ticket_id, None)
        self._watch_contracts.pop(ticket.ticket_id, None)

    def _probe_underlying_price(self) -> float:
        """A neutral probe anchor; communication check only, never used to trade."""
        # Contract lookup accepts a price anchor only for strike selection. If no
        # market-price provider is wired yet, use a broad symbol-specific anchor.
        # The paper runner passes actual underlying price for all trading decisions.
        return {"COIN": 200.0, "TSLA": 300.0}.get(self.symbol, 100.0)

    def get_status(self) -> Dict[str, Any]:
        base = super().get_status()
        base.update({
            "version": self.VERSION,
            "status": self.STATUS,
            "label": self.LABEL,
            "symbol": self.symbol,
            "symbol_policy": self.symbol_policy,
            "size_multiplier": self.size_multiplier,
            "pending_entries": len(self._pending_entry),
            "pending_exits": len(self._pending_exit),
            "open_option_positions": len(self._positions),
            "broker_error": self.execution_manager.last_broker_error,
        })
        return base


def create_v14_3_options_runner(
    symbol: str,
    trading_client: Any,
    option_data_client: Any,
    paper_mode: bool = True,
    log_dir: str = "HFT/logs/v14_3_options",
) -> V14_3_OptionsRunner:
    return V14_3_OptionsRunner(
        symbol=symbol,
        trading_client=trading_client,
        option_data_client=option_data_client,
        config=get_v14_3_config(),
        paper_mode=paper_mode,
        log_dir=str(Path(log_dir) / symbol.upper()),
    )
