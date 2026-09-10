"""V14.5.1 validation-hardened options trader.

This is a hardening pass, not a new directional strategy.  It preserves V14.5's
CALL/PUT hysteresis and core/runner payoff design while closing validation and
execution gaps found in the end-to-end review.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from execution.mleg_execution_manager import MlegLeg, OrderState
from execution.v14_5_1_mleg_execution_manager import V14_5_1_MlegExecutionManager
from strategy.confirmation_engine import ConfirmationEngine, ConfirmationResult
from strategy.empirical_option_edge import EmpiricalOptionEdgeMemory
from strategy.options_reversal_guard import GuardDecision, OptionsReversalGuard
from strategy.v14_3_highvol_config import can_trade_symbol
from strategy.v14_3_options_runner import V14_3_OptionsRunner, PositionPlan
from strategy.v14_5_stable_options import V14_5_OptionsTrader, _NoPyramidConditioner
from strategy.v14_5_1_config import get_v14_5_1_config
from strategy.v14_5_1_vertical_selector import RankedVerticalSelector
from strategy.watch_ticket import TicketSide, TicketStatus


class BarAwareOptionsReversalGuard(OptionsReversalGuard):
    """Advance hysteresis only once per unique market bar/timestamp."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._last_observation: Dict[str, str] = {}

    def select_candidates(self, symbol: str, candidates: List, *, observation_id=None, **kwargs):
        key = str(observation_id or "")
        sym = symbol.upper()
        st = self.state(sym)
        if key and self.cfg.get("direction_unique_bar_only", True) and self._last_observation.get(sym) == key:
            call, put = self._top_by_side(candidates)
            call_score = float(call.score) if call is not None else 0.0
            put_score = float(put.score) if put is not None else 0.0
            allowed = [c for c in candidates if str(c.side).upper() == st.bias] if st.bias in ("CALL", "PUT") else []
            return GuardDecision(
                allowed_candidates=allowed,
                bias=st.bias,
                proposed_side=("CALL" if call_score >= put_score and call_score > 0 else ("PUT" if put_score > 0 else "NEUTRAL")),
                reason="duplicate_market_bar_ignored",
                top_call_score=call_score,
                top_put_score=put_score,
                score_edge=abs(call_score - put_score),
                flip_streak=st.flip_streak,
                size_multiplier=self._size_multiplier(st),
            )
        if key:
            self._last_observation[sym] = key
        return super().select_candidates(symbol, candidates, **kwargs)


class HardenedRouteConditioner:
    """Route score -> empirical edge annotation -> bar-aware hysteresis."""

    def __init__(self, base, guard: BarAwareOptionsReversalGuard, edge_memory: EmpiricalOptionEdgeMemory, owner):
        self.base = base
        self.guard = guard
        self.edge_memory = edge_memory
        self.owner = owner

    def evaluate_routes(self, **kwargs):
        raw = self.base.evaluate_routes(**kwargs)
        regime = self.owner._current_regime()
        self.owner._latest_route_scores = {}
        for candidate in raw:
            self.owner._latest_route_scores[(str(candidate.side).upper(), candidate.route)] = float(candidate.score)
            self.edge_memory.annotate_candidate(candidate, self.owner.symbol, regime)

        open_side, has_pending, conflict = self.owner._directional_risk_state()
        if conflict:
            self.owner._last_guard_decision = GuardDecision(
                allowed_candidates=[], bias="CONFLICT", reason="multiple_directional_risk_sides"
            )
            return []
        decision = self.guard.select_candidates(
            self.owner.symbol,
            raw,
            price=float(kwargs.get("price", 0.0)),
            vwap=float(kwargs.get("vwap", 0.0)),
            atr=float(kwargs.get("atr", 0.0)),
            open_risk_side=open_side,
            has_pending_risk=has_pending,
            observation_id=self.owner._market_observation_id,
        )
        self.owner._last_guard_decision = decision
        if decision.bias_changed and self.owner.cfg.get("cancel_opposite_watches_on_bias_lock", True):
            self.owner._retire_opposite_watches(decision.bias)
        return decision.allowed_candidates


class LiveRouteConfirmationEngine(ConfirmationEngine):
    """Use the route's CURRENT score for the decay test."""

    def __init__(self, owner, config: dict):
        super().__init__(config=config)
        self.owner = owner

    def check_confirmation(self, ticket, *args, route_score_now=None, **kwargs):
        side = "CALL" if ticket.side == TicketSide.CALL else "PUT"
        current = float(self.owner._latest_route_scores.get((side, ticket.route), 0.0))
        floor = max(
            float(self.cfg["watch_min_route_score"]),
            float(ticket.route_score) * float(self.cfg.get("route_decay_floor_ratio", 0.90)),
        )
        if current < floor:
            return ConfirmationResult(
                confirmed=False,
                reason=f"route_decayed_live:{current:.4f}<{floor:.4f}",
                current_price=float(kwargs.get("current_price", args[0] if args else 0.0)),
                mfe_velocity=float(kwargs.get("mfe_velocity", 0.0)),
                env_stress=float(kwargs.get("env_stress", 0.0)),
            )
        return super().check_confirmation(ticket, *args, route_score_now=current, **kwargs)


class V14_5_1_OptionsTrader(V14_5_OptionsTrader):
    VERSION = "14.5.1-hardened-options"
    STATUS = "VALIDATION_HARDENED_PAPER_REPLAY"
    LABEL = "V14_5_1_VALIDATION_HARDENED_OPTIONS"

    def __init__(
        self,
        symbol: str,
        trading_client: Any,
        option_data_client: Any,
        config: Optional[dict] = None,
        paper_mode: bool = True,
        log_dir: str = "HFT/logs/v14_5_1",
        allow_offline_simulation: bool = False,
    ):
        cfg = config or get_v14_5_1_config()
        super().__init__(
            symbol=symbol,
            trading_client=trading_client,
            option_data_client=option_data_client,
            config=cfg,
            paper_mode=paper_mode,
            log_dir=log_dir,
            allow_offline_simulation=allow_offline_simulation,
        )
        self.execution_manager = V14_5_1_MlegExecutionManager(
            trading_client=trading_client,
            config=self.cfg,
            log_dir=str(self.log_dir),
            paper_mode=paper_mode,
            allow_offline_simulation=allow_offline_simulation,
        )
        self.edge_memory = EmpiricalOptionEdgeMemory(
            min_samples=self.cfg["empirical_edge_min_samples"],
            window=self.cfg["empirical_edge_window"],
            prior_strength=self.cfg["empirical_edge_prior_strength"],
            log_path=str(Path(self.log_dir) / "empirical_option_edge.csv"),
        )
        self.reversal_guard = BarAwareOptionsReversalGuard(self.cfg)
        # Bypass the old GuardedRouteConditioner and rebuild from the raw V14 route
        # engine saved by V14.5's constructor.
        hardened = HardenedRouteConditioner(
            self._base_route_conditioner, self.reversal_guard, self.edge_memory, self
        )
        self.route_conditioner = _NoPyramidConditioner(hardened, self)
        self.confirmation_engine = LiveRouteConfirmationEngine(self, self.cfg)
        if trading_client is not None and option_data_client is not None:
            self.chain_fetcher = RankedVerticalSelector(trading_client, option_data_client, self.cfg)

        self._market_observation_id = ""
        self._latest_route_scores: Dict[Tuple[str, str], float] = {}
        self._latest_iv_percentile = 0.5
        self._emergency_exit_required: Dict[str, str] = {}

    def _current_regime(self) -> str:
        iv = float(self._latest_iv_percentile)
        if iv >= 0.70:
            return "HIGH_IV"
        if iv <= 0.30:
            return "LOW_IV"
        return "NORMAL_IV"

    def evaluate_opportunity(self, *args, market_timestamp=None, **kwargs):
        self._market_observation_id = str(market_timestamp or datetime.utcnow().replace(second=0, microsecond=0).isoformat())
        self._latest_iv_percentile = float(kwargs.get("iv_percentile", 0.5) or 0.5)
        return super().evaluate_opportunity(*args, **kwargs)

    def _passes_v14_3_watch(self, candidate) -> bool:
        if candidate.route not in (self.cfg["routes_allowed"] + self.cfg["routes_soft_only"]):
            return False
        if float(candidate.score) < float(self.cfg["watch_min_route_score"]):
            return False

        # Never reuse route-score-derived pseudo IC/EV.  Mature empirical evidence
        # must pass independent gates.  Before maturity, PAPER/REPLAY may collect
        # labels at a size haircut; live is disabled by config anyway.
        ready = bool(getattr(candidate, "empirical_edge_ready", False))
        if not ready:
            return bool(self.paper_mode)
        ic = float(getattr(candidate, "empirical_ic_spread", 0.0))
        ev = float(getattr(candidate, "empirical_ev_over_debit", 0.0))
        if self.cfg.get("empirical_negative_edge_blocks", True) and (ic <= 0 or ev <= 0):
            return False
        return (
            ic >= float(self.cfg["empirical_min_ic_spread"])
            and ev >= float(self.cfg["empirical_min_ev_over_debit"])
        )

    def _attempt_real_execution(self, ticket, underlying_price: float, iv_percentile: float):
        allowed, reason = self.reversal_guard.ticket_side_allowed(ticket.symbol, self._side_name(ticket.side))
        if not allowed:
            ticket.mark_rejected(f"reversal_guard:{reason}")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            self._watch_contracts.pop(ticket.ticket_id, None)
            return {"action": "REJECTED", "reason": reason}

        base_allowed, _, base_size = can_trade_symbol(ticket.symbol)
        if not base_allowed:
            return {"action": "REJECTED", "reason": "symbol_policy_blocked"}

        st = self.reversal_guard.state(ticket.symbol)
        if st.cooldown_remaining > 0:
            stability_mult = float(self.cfg.get("recent_reversal_size_multiplier", 0.50))
        elif st.bars_in_bias < int(self.cfg.get("stable_bias_full_size_after_bars", 4)):
            stability_mult = float(self.cfg.get("new_bias_size_multiplier", 0.75))
        else:
            stability_mult = 1.0

        edge = self.edge_memory.stats(ticket.symbol, ticket.route, self._current_regime())
        edge_mult = 1.0 if edge.ready else float(self.cfg.get("empirical_unvalidated_size_multiplier", 0.50))

        # Preflight the exact frozen watch contracts.  This both enforces the next-
        # quote rule and makes the missing-Greeks haircut real rather than a comment.
        fresh = self._fresh_exact_quotes(ticket)
        if fresh is None:
            return {"action": "WAIT_NEXT_QUOTE", "reason": "waiting_for_quote_newer_than_confirmation"}
        long_leg, short_leg, _, _ = fresh
        greeks_available = (
            abs(float(getattr(long_leg, "delta", 0.0) or 0.0)) > 0
            and abs(float(getattr(short_leg, "delta", 0.0) or 0.0)) > 0
        )
        greek_mult = 1.0 if greeks_available else float(self.cfg.get("option_missing_greeks_size_multiplier", 0.65))

        self.size_multiplier = float(base_size) * stability_mult * edge_mult * greek_mult
        result = V14_3_OptionsRunner._attempt_real_execution(self, ticket, underlying_price, iv_percentile)
        result.setdefault("size_multiplier", self.size_multiplier)
        result["empirical_edge_ready"] = edge.ready
        result["empirical_edge_samples"] = edge.samples
        result["greeks_available"] = greeks_available
        return result

    # ------------------------------------------------------------------
    # Partial entry reconciliation
    # ------------------------------------------------------------------
    def _allocation_for_actual_qty(self, pending: dict, qty: int) -> Tuple[str, int, int, int]:
        if pending["package"].is_package and qty >= 2:
            core = max(1, round(qty * float(self.cfg["core_fraction"])))
            runner = max(1, qty - core)
            if core + runner > qty:
                core = qty - runner
            return "PACKAGE", core, runner, 0
        return "FALLBACK", 0, 0, qty

    def _sync_partial_entry(self, ticket_id: str, actual_qty: int) -> None:
        pending = self._pending_entry.get(ticket_id)
        ticket = self.ticket_book.active_tickets.get(ticket_id)
        if not pending or not ticket or actual_qty <= 0:
            return
        order = pending["order"]
        actual_qty = min(int(actual_qty), int(pending["qty"]))
        applied = int(pending.get("activated_qty", 0))
        if actual_qty <= applied:
            return
        entry_debit = float(
            getattr(order, "broker_filled_avg_price", 0.0)
            or order.actual_fill_price
            or order.intended_limit
        )
        mode, core_qty, runner_qty, fallback_qty = self._allocation_for_actual_qty(pending, actual_qty)
        plan = self._positions.get(ticket_id)
        if plan is None:
            plan = PositionPlan(
                ticket_id=ticket_id,
                symbol=ticket.symbol,
                mode=mode,
                long_contract=pending["long_contract"],
                short_contract=pending["short_contract"],
                entry_order_id=order.order_id,
                total_qty=actual_qty,
                core_qty=core_qty,
                runner_qty=runner_qty,
                fallback_qty=fallback_qty,
                entry_debit=entry_debit,
                total_debit_dollars=entry_debit * 100.0 * actual_qty,
            )
            plan.initial_debit_dollars = plan.total_debit_dollars
            self._positions[ticket_id] = plan
            ticket.mark_filled()
            self.risk_manager.add_open_position(ticket.symbol, plan.total_debit_dollars)
            self.risk_manager.record_trade(ticket.symbol, ticket.route, plan.total_debit_dollars)
            self.slippage_monitor.record_entry_fill(order.fill_slippage_vs_mid, was_filled=True)
        else:
            plan.mode = mode
            plan.total_qty = actual_qty
            plan.core_qty = core_qty
            plan.runner_qty = runner_qty
            plan.fallback_qty = fallback_qty
            plan.entry_debit = entry_debit
            plan.total_debit_dollars = entry_debit * 100.0 * actual_qty
            # One strategy per symbol allows exact replacement of exposure here.
            self.risk_manager._open_positions[ticket.symbol] = plan.total_debit_dollars
            plan.initial_debit_dollars = max(
                float(getattr(plan, "initial_debit_dollars", 0.0)), plan.total_debit_dollars
            )
        pending["activated_qty"] = actual_qty

    # ------------------------------------------------------------------
    # Exit reconciliation / bounded cancel-reprice
    # ------------------------------------------------------------------
    @staticmethod
    def _hard_exit_reason(reason: str) -> bool:
        text = str(reason).lower()
        return "stop" in text or "runner_lock" in text or "risk" in text

    def _submit_partial_exit(self, plan: PositionPlan, qty: int, close_bid: float, close_ask: float, reason: str, role: str, retry_count: int = 0):
        qty = min(int(qty), int(plan.total_qty))
        if qty <= 0:
            return
        concession = min(
            float(self.cfg.get("hard_exit_max_concession_pct", 0.12)),
            retry_count * float(self.cfg.get("hard_exit_reprice_step_pct", 0.015)),
        ) if self._hard_exit_reason(reason) else 0.0
        credit = max(0.01, float(close_bid) * (1.0 - concession))
        legs = [
            MlegLeg(plan.long_contract.symbol, "sell_to_close", qty, plan.long_contract.option_type, plan.long_contract.strike, plan.long_contract.expiration),
            MlegLeg(plan.short_contract.symbol, "buy_to_close", qty, plan.short_contract.option_type, plan.short_contract.strike, plan.short_contract.expiration),
        ]
        order = self.execution_manager.create_exit_order(
            ticket_id=plan.ticket_id,
            legs=legs,
            limit_price=-credit,
            quantity=qty,
            exit_bid=float(close_bid),
            exit_ask=float(close_ask),
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
            "applied_qty": 0,
            "retry_count": retry_count,
            "cancel_requested": False,
        }
        if not self.execution_manager.submit_exit(order):
            self._pending_exit.pop(order.order_id, None)
            return
        if order.state == OrderState.CLOSED:
            order.broker_filled_qty = qty
            self._apply_exit_fill_delta(order.order_id, qty)

    def _apply_exit_fill_delta(self, order_id: str, filled_qty: int) -> None:
        meta = self._pending_exit.get(order_id)
        if not meta:
            return
        plan = self._positions.get(meta["ticket_id"])
        ticket = self.ticket_book.active_tickets.get(meta["ticket_id"])
        if not plan or not ticket:
            return
        applied = int(meta.get("applied_qty", 0))
        target = min(int(filled_qty), int(meta["qty"]))
        delta = target - applied
        if delta <= 0:
            return
        order = meta["order"]
        credit = abs(float(order.actual_exit_price or meta["expected_credit"]))
        plan.realized_pnl += (credit - plan.entry_debit) * 100.0 * delta
        plan.total_qty = max(0, int(plan.total_qty) - delta)

        role = meta["role"]
        if role == "CORE":
            plan.core_qty = max(0, int(plan.core_qty) - delta)
            if plan.core_qty == 0:
                plan.core_closed = True
                plan.runner_lock_active = True
        elif role == "RUNNER":
            plan.runner_qty = max(0, int(plan.runner_qty) - delta)
        else:  # ALL
            if plan.mode == "FALLBACK":
                plan.fallback_qty = max(0, int(plan.fallback_qty) - delta)
            else:
                take_core = min(delta, int(plan.core_qty))
                plan.core_qty -= take_core
                remaining = delta - take_core
                if remaining > 0:
                    plan.runner_qty = max(0, int(plan.runner_qty) - remaining)

        meta["applied_qty"] = target
        self.risk_manager._open_positions[ticket.symbol] = max(0.0, plan.entry_debit * 100.0 * plan.total_qty)
        self.slippage_monitor.record_exit_fill(order.exit_slippage_vs_bid)

        if plan.total_qty > 0:
            return

        reason = str(meta["reason"])
        initial_debit = max(float(getattr(plan, "initial_debit_dollars", 0.0)), 1e-9)
        spread_return = float(plan.realized_pnl) / initial_debit
        self.edge_memory.record(
            symbol=ticket.symbol,
            route=ticket.route,
            route_score=ticket.route_score,
            spread_return=spread_return,
            regime=self._current_regime(),
            source="PAPER_FILL" if self.paper_mode else "LIVE_FILL",
        )
        ticket.mark_closed(plan.realized_pnl, reason)
        self.ticket_book.retire_ticket(ticket.ticket_id)
        self.risk_manager.record_close(ticket.symbol, plan.realized_pnl)
        self.route_monitor.record_trade(ticket.route, plan.realized_pnl)
        self.env_monitor.record_trade(self._current_environment, ticket.symbol, plan.realized_pnl)
        self.reversal_guard.record_exit(ticket.symbol, self._side_name(ticket.side), reason)
        self._positions.pop(ticket.ticket_id, None)
        self._watch_contracts.pop(ticket.ticket_id, None)
        self._exit_structure_fail_streak.pop(ticket.ticket_id, None)

    def _request_stale_exit_action(self, order_id: str, meta: dict) -> None:
        order = meta["order"]
        if not order.submit_time or meta.get("cancel_requested"):
            return
        elapsed = (datetime.utcnow() - order.submit_time).total_seconds()
        hard = self._hard_exit_reason(meta["reason"])
        threshold = float(self.cfg["hard_exit_requote_seconds"] if hard else self.cfg["soft_exit_requote_seconds"])
        if elapsed < threshold:
            return
        if hard and int(meta.get("retry_count", 0)) >= int(self.cfg["hard_exit_max_requotes"]):
            self.execution_manager.mark_emergency_exit_required(order, "hard_exit_requote_exhausted")
            self._emergency_exit_required[meta["ticket_id"]] = str(meta["reason"])
            return
        if self.execution_manager.cancel_remainder(order, "stale_exit_reprice" if hard else "stale_soft_exit_cancel"):
            meta["cancel_requested"] = True

    def _resubmit_remaining_exit(self, meta: dict) -> None:
        ticket_id = meta["ticket_id"]
        plan = self._positions.get(ticket_id)
        if not plan:
            return
        remaining = min(int(meta["qty"]) - int(meta.get("applied_qty", 0)), int(plan.total_qty))
        if remaining <= 0:
            return
        if not self._hard_exit_reason(meta["reason"]):
            # Target/continuation exits are re-evaluated from fresh strategy state;
            # do not chase a stale profit condition.
            return
        quote = self._current_spread_quote(plan)
        if quote is None:
            self._emergency_exit_required[ticket_id] = "hard_exit_quote_unavailable"
            return
        _, _, close_bid, close_ask, _ = quote
        self._submit_partial_exit(
            plan, remaining, close_bid, close_ask,
            meta["reason"], meta["role"], retry_count=int(meta.get("retry_count", 0)) + 1,
        )

    def poll_broker_orders(self):
        # Entries: activate every actually filled strategy unit before interpreting
        # the parent order's terminal status.
        for ticket_id, pending in list(self._pending_entry.items()):
            order = pending["order"]
            self.execution_manager.check_order_status(order)
            filled = int(float(getattr(order, "broker_filled_qty", 0.0) or 0.0))
            if filled > int(pending.get("activated_qty", 0)):
                self._sync_partial_entry(ticket_id, filled)

            if order.state == OrderState.PARTIALLY_FILLED and order.submit_time:
                elapsed = (datetime.utcnow() - order.submit_time).total_seconds()
                if elapsed >= float(self.cfg["partial_fill_cancel_after_seconds"]):
                    self.execution_manager.cancel_remainder(order, "partial_entry_exposure_exists")
            if order.state == OrderState.FILLED:
                self._sync_partial_entry(ticket_id, int(pending["qty"]))
                self._pending_entry.pop(ticket_id, None)
            elif order.state in (OrderState.REJECTED, OrderState.CANCELED, OrderState.EXPIRED, OrderState.MISSED_FILL):
                activated = int(pending.get("activated_qty", 0))
                if activated <= 0:
                    ticket = self.ticket_book.active_tickets.get(ticket_id)
                    if ticket:
                        ticket.mark_missed_fill(order.cancel_reason or order.missed_fill_reason or order.state.value)
                        self.ticket_book.retire_ticket(ticket_id)
                    self.slippage_monitor.record_entry_fill(0.0, was_filled=False)
                # If activated > 0, the position stays live and managed.  Only the
                # unfilled remainder is retired.
                self._pending_entry.pop(ticket_id, None)

        # Exits: apply fill deltas immediately; never wait for the full parent qty
        # before reducing strategy exposure.
        for order_id, meta in list(self._pending_exit.items()):
            order = meta["order"]
            self.execution_manager.check_order_status(order)
            filled = int(float(getattr(order, "broker_filled_qty", 0.0) or 0.0))
            if filled > int(meta.get("applied_qty", 0)):
                self._apply_exit_fill_delta(order_id, filled)
            if order_id not in self._pending_exit:
                continue

            if order.state == OrderState.CLOSED:
                self._pending_exit.pop(order_id, None)
                continue
            if order.state in (OrderState.CANCELED, OrderState.EXPIRED, OrderState.REJECTED, OrderState.CLOSE_FAILED):
                old = self._pending_exit.pop(order_id, None)
                if old:
                    self._resubmit_remaining_exit(old)
                continue
            self._request_stale_exit_action(order_id, meta)

    def get_status(self):
        out = super().get_status()
        out.update({
            "version": self.VERSION,
            "status": self.STATUS,
            "label": self.LABEL,
            "market_observation_id": self._market_observation_id,
            "empirical_edge_regime": self._current_regime(),
            "max_open_debit_exposure_pct": float(self.cfg["max_open_debit_exposure_pct"]),
            "emergency_exit_required": dict(self._emergency_exit_required),
            "selector_candidates": list(getattr(self.chain_fetcher, "last_ranked_candidates", [])),
        })
        return out


def create_v14_5_1_options_trader(
    symbol: str,
    trading_client: Any,
    option_data_client: Any,
    paper_mode: bool = True,
    log_dir: str = "HFT/logs/v14_5_1",
) -> V14_5_1_OptionsTrader:
    return V14_5_1_OptionsTrader(
        symbol=symbol,
        trading_client=trading_client,
        option_data_client=option_data_client,
        config=get_v14_5_1_config(),
        paper_mode=paper_mode,
        log_dir=str(Path(log_dir) / symbol.upper()),
    )
