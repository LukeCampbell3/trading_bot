"""V14.5 stable-direction options runner.

This is an execution-compatible extension of V14.3, not a new payoff model.
It preserves the V14.3 watch -> confirmation -> real spread quality -> MLEG
core/runner pipeline while adding:

1. CALL/PUT directional hysteresis so one noisy bar cannot reverse the book.
2. A hard directional lock while filled/confirmed/pending option risk exists.
3. Stronger, persistent evidence for an actual reversal than for an initial bias.
4. Post-exit reversal cooldown, longer after stops/losses.
5. Option snapshot enrichment (IV + Greeks when Alpaca supplies them).
6. Option-native delta/theta/IV-skew quality checks before MLEG submission.
7. Stability-aware debit sizing for new/recently reversed directional states.

Live trading remains disabled by configuration.  V14.5 is a paper/replay
validation candidate until broker-backed evidence proves the new filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from strategy.v14_3_options_runner import V14_3_OptionsRunner
from strategy.v14_3_highvol_config import can_trade_symbol, get_symbol_policy
from strategy.v14_5_stability_config import get_v14_5_config
from strategy.options_reversal_guard import GuardDecision, OptionsReversalGuard
from strategy.option_chain_fetcher import OptionChainFetcher, ContractInfo
from strategy.spread_quality_gate import OptionLeg, SpreadQualityGate, SpreadQualityReport
from strategy.watch_ticket import TicketSide, TicketStatus

try:
    from alpaca.data.requests import OptionSnapshotRequest
    _SNAPSHOT_SDK_OK = True
except Exception:  # pragma: no cover - depends on installed alpaca-py
    OptionSnapshotRequest = None
    _SNAPSHOT_SDK_OK = False


class SnapshotEnrichedOptionChainFetcher(OptionChainFetcher):
    """Use Alpaca option snapshots when available, with quote-only fallback.

    Alpaca snapshots expose latest quote, implied volatility and Greeks for a
    contract.  V14.3 only consumed latest quotes, which meant delta/IV fields in
    OptionLeg were normally zero.  This wrapper keeps the same public interface
    while populating option-native metrics defensively.
    """

    @staticmethod
    def _num(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _leg_from_snapshot(self, contract: ContractInfo, side: str, snap: Any) -> Optional[OptionLeg]:
        quote = getattr(snap, "latest_quote", None)
        if quote is None:
            return None
        bid = self._num(getattr(quote, "bid_price", 0.0))
        ask = self._num(getattr(quote, "ask_price", 0.0))
        greeks = getattr(snap, "greeks", None)
        iv = self._num(getattr(snap, "implied_volatility", 0.0))
        leg = OptionLeg(
            contract_symbol=contract.symbol,
            side=side,
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2.0 if (bid + ask) > 0 else 0.0,
            delta=self._num(getattr(greeks, "delta", 0.0)) if greeks else 0.0,
            iv=iv,
            dte=(date.fromisoformat(contract.expiration) - date.today()).days,
            strike=contract.strike,
            quote_timestamp=str(getattr(quote, "timestamp", "")),
        )
        # OptionLeg intentionally stays backwards-compatible.  Extra Greeks are
        # attached dynamically for the V14.5 native quality gate.
        leg.gamma = self._num(getattr(greeks, "gamma", 0.0)) if greeks else 0.0
        leg.theta = self._num(getattr(greeks, "theta", 0.0)) if greeks else 0.0
        leg.vega = self._num(getattr(greeks, "vega", 0.0)) if greeks else 0.0
        return leg

    def get_live_quotes(
        self, long_contract: ContractInfo, short_contract: ContractInfo
    ) -> Optional[Tuple[OptionLeg, OptionLeg]]:
        if _SNAPSHOT_SDK_OK and hasattr(self.data_client, "get_option_snapshot"):
            try:
                req = OptionSnapshotRequest(
                    symbol_or_symbols=[long_contract.symbol, short_contract.symbol]
                )
                snaps = self.data_client.get_option_snapshot(req)
                if snaps:
                    long_snap = snaps.get(long_contract.symbol)
                    short_snap = snaps.get(short_contract.symbol)
                    if long_snap is not None and short_snap is not None:
                        long_leg = self._leg_from_snapshot(long_contract, "buy", long_snap)
                        short_leg = self._leg_from_snapshot(short_contract, "sell", short_snap)
                        if long_leg is not None and short_leg is not None:
                            return long_leg, short_leg
            except Exception as exc:  # quote-only fallback is intentional
                print(f"[V14.5 SNAPSHOT] snapshot enrichment unavailable: {exc}")
        return super().get_live_quotes(long_contract, short_contract)


class OptionNativeSpreadQualityGate(SpreadQualityGate):
    """Augment V14.3 spread-quality checks with option-native stability metrics."""

    def evaluate(self, *args, **kwargs) -> SpreadQualityReport:
        report = super().evaluate(*args, **kwargs)
        if not report.passed:
            return report

        long_leg: OptionLeg = kwargs.get("long_leg") if "long_leg" in kwargs else args[0]
        short_leg: OptionLeg = kwargs.get("short_leg") if "short_leg" in kwargs else args[1]

        long_delta = abs(float(getattr(long_leg, "delta", 0.0) or 0.0))
        short_delta = abs(float(getattr(short_leg, "delta", 0.0) or 0.0))
        long_theta = float(getattr(long_leg, "theta", 0.0) or 0.0)
        short_theta = float(getattr(short_leg, "theta", 0.0) or 0.0)
        long_iv = float(getattr(long_leg, "iv", 0.0) or 0.0)
        short_iv = float(getattr(short_leg, "iv", 0.0) or 0.0)

        greeks_available = long_delta > 0 and short_delta > 0
        delta_spread = long_delta - short_delta if greeks_available else 0.0
        net_theta = long_theta - short_theta
        theta_burden = (
            abs(min(0.0, net_theta)) / report.spread_mid
            if report.spread_mid > 0 and (long_theta != 0 or short_theta != 0)
            else 0.0
        )
        iv_skew = abs(long_iv - short_iv) if long_iv > 0 and short_iv > 0 else 0.0

        # Add diagnostics without changing the base report type/API.
        report.greeks_available = greeks_available
        report.short_leg_delta = short_delta
        report.delta_spread_abs = delta_spread
        report.net_theta = net_theta
        report.theta_burden_pct_of_mid = theta_burden
        report.iv_skew = iv_skew

        if not greeks_available:
            # Missing Greeks are not silently treated as ideal.  Execution remains
            # possible because indicative/data entitlements can vary, but V14.5
            # applies a size haircut in the runner.
            report.option_native_quality_score = 0.50
            return report

        dmin = float(self.cfg.get("option_long_delta_min_abs", 0.40))
        dmax = float(self.cfg.get("option_long_delta_max_abs", 0.72))
        if not (dmin <= long_delta <= dmax):
            report.passed = False
            report.rejection_reason = f"long_delta_out_of_band:{long_delta:.4f}"
            return report

        min_delta_spread = float(self.cfg.get("option_min_delta_spread_abs", 0.07))
        if delta_spread < min_delta_spread:
            report.passed = False
            report.rejection_reason = f"delta_spread_too_small:{delta_spread:.4f}"
            return report

        max_theta = float(self.cfg.get("option_max_theta_burden_pct_of_mid", 0.12))
        if theta_burden > max_theta:
            report.passed = False
            report.rejection_reason = f"theta_burden_exceeded:{theta_burden:.4f}"
            return report

        max_iv_skew = float(self.cfg.get("option_max_leg_iv_skew", 0.15))
        if iv_skew > max_iv_skew:
            report.passed = False
            report.rejection_reason = f"leg_iv_skew_exceeded:{iv_skew:.4f}"
            return report

        target = float(self.cfg.get("option_delta_target_abs", 0.55))
        delta_fit = max(0.0, 1.0 - abs(long_delta - target) / 0.25)
        delta_sep_score = max(0.0, min(1.0, delta_spread / 0.20))
        theta_score = 1.0 if max_theta <= 0 else max(0.0, 1.0 - theta_burden / max_theta)
        iv_score = 1.0 if max_iv_skew <= 0 else max(0.0, 1.0 - iv_skew / max_iv_skew)
        width_cap = max(float(self.cfg.get("max_composite_spread_pct_of_mid", 0.15)), 1e-9)
        width_score = max(0.0, 1.0 - report.composite_spread_pct_of_mid / width_cap)
        native_score = (
            0.30 * delta_fit
            + 0.20 * delta_sep_score
            + 0.15 * theta_score
            + 0.10 * iv_score
            + 0.25 * width_score
        )
        report.option_native_quality_score = native_score
        if native_score < float(self.cfg.get("option_min_native_quality_score", 0.55)):
            report.passed = False
            report.rejection_reason = f"option_native_quality_low:{native_score:.4f}"
        return report


class GuardedRouteConditioner:
    """Adapter that applies directional hysteresis to the existing route scores."""

    def __init__(self, base, guard: OptionsReversalGuard, owner: "V14_5_StableOptionsRunner"):
        self.base = base
        self.guard = guard
        self.owner = owner

    def evaluate_routes(self, **kwargs):
        raw = self.base.evaluate_routes(**kwargs)
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
        )
        self.owner._last_guard_decision = decision
        if decision.bias_changed and self.owner.cfg.get("cancel_opposite_watches_on_bias_lock", True):
            self.owner._retire_opposite_watches(decision.bias)
        return decision.allowed_candidates


class V14_5_StableOptionsRunner(V14_3_OptionsRunner):
    VERSION = "14.5-options-stable-direction"
    STATUS = "STABLE_DIRECTION_OPTIONS_PAPER_VALIDATION"
    LABEL = "V14_5_STABLE_OPTIONS_REVERSAL_CONTROL"

    def __init__(
        self,
        symbol: str,
        trading_client: Any,
        option_data_client: Any,
        config: Optional[dict] = None,
        paper_mode: bool = True,
        log_dir: str = "HFT/logs/v14_5_options",
        allow_offline_simulation: bool = False,
    ):
        cfg = config or get_v14_5_config()
        super().__init__(
            symbol=symbol,
            trading_client=trading_client,
            option_data_client=option_data_client,
            config=cfg,
            paper_mode=paper_mode,
            log_dir=log_dir,
            allow_offline_simulation=allow_offline_simulation,
        )
        self.reversal_guard = OptionsReversalGuard(self.cfg)
        self._base_route_conditioner = self.route_conditioner
        self.route_conditioner = GuardedRouteConditioner(
            self._base_route_conditioner, self.reversal_guard, self
        )
        self.spread_gate = OptionNativeSpreadQualityGate(config=self.cfg)
        if trading_client is not None and option_data_client is not None:
            self.chain_fetcher = SnapshotEnrichedOptionChainFetcher(
                trading_client, option_data_client
            )
        self._last_guard_decision = GuardDecision(reason="not_evaluated")
        self._symbol_policy_size_multiplier = float(self.size_multiplier)

    @staticmethod
    def _side_name(value: Any) -> str:
        text = str(value or "").upper()
        if text.endswith(".CALL"):
            return "CALL"
        if text.endswith(".PUT"):
            return "PUT"
        return text if text in ("CALL", "PUT") else "NEUTRAL"

    def _directional_risk_state(self) -> Tuple[str, bool, bool]:
        """Return (filled/confirmed side, has pending risk, side conflict)."""
        sides = set()
        for ticket_id in self._positions:
            ticket = self.ticket_book.active_tickets.get(ticket_id)
            if ticket:
                side = self._side_name(ticket.side)
                if side in ("CALL", "PUT"):
                    sides.add(side)
        for ticket in self.ticket_book.active_tickets.values():
            if ticket.symbol != self.symbol:
                continue
            if ticket.status in (TicketStatus.CONFIRMED, TicketStatus.FILLED):
                side = self._side_name(ticket.side)
                if side in ("CALL", "PUT"):
                    sides.add(side)
        pending = bool(self._pending_entry or self._pending_exit)
        if len(sides) > 1:
            return "CONFLICT", pending, True
        side = next(iter(sides), "NEUTRAL")
        return side, pending, False

    def _retire_opposite_watches(self, allowed_side: str) -> None:
        allowed_side = self._side_name(allowed_side)
        for ticket in list(self.ticket_book.active_tickets.values()):
            if ticket.symbol != self.symbol or ticket.status != TicketStatus.WATCHING:
                continue
            if self._side_name(ticket.side) == allowed_side:
                continue
            ticket.mark_rejected(f"direction_bias_locked:{allowed_side}")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            self._watch_contracts.pop(ticket.ticket_id, None)
            self._watch_high.pop(ticket.ticket_id, None)
            self._watch_low.pop(ticket.ticket_id, None)

    def _attempt_real_execution(self, ticket, underlying_price: float, iv_percentile: float):
        allowed, reason = self.reversal_guard.ticket_side_allowed(
            ticket.symbol, self._side_name(ticket.side)
        )
        if not allowed:
            ticket.mark_rejected(f"reversal_guard:{reason}")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            self._watch_contracts.pop(ticket.ticket_id, None)
            return {"action": "REJECTED", "reason": reason}

        # Base symbol policy x stability multiplier.  Missing Greeks receive an
        # additional haircut later through the package affordability path only
        # when snapshot enrichment is unavailable.
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
        self.size_multiplier = float(base_size) * stability_mult
        return super()._attempt_real_execution(ticket, underlying_price, iv_percentile)

    def _finalize_exit(self, order_id: str):
        meta = self._pending_exit.get(order_id)
        if not meta:
            return super()._finalize_exit(order_id)
        ticket = self.ticket_book.active_tickets.get(meta.get("ticket_id"))
        side = self._side_name(ticket.side) if ticket else "NEUTRAL"
        symbol = ticket.symbol if ticket else self.symbol
        role = meta.get("role")
        reason = str(meta.get("reason", ""))
        super()._finalize_exit(order_id)
        # CORE partial profit does not flatten directional exposure.  ALL/RUNNER
        # exits do, so they start the re-entry/reversal hysteresis clock.
        if role in ("ALL", "RUNNER"):
            self.reversal_guard.record_exit(symbol, side, reason)

    def get_status(self) -> Dict[str, Any]:
        base = super().get_status()
        st = self.reversal_guard.state(self.symbol)
        d = self._last_guard_decision
        base.update({
            "version": self.VERSION,
            "status": self.STATUS,
            "label": self.LABEL,
            "direction_bias": st.bias,
            "bars_in_direction_bias": st.bars_in_bias,
            "reversal_flip_streak": st.flip_streak,
            "reversal_cooldown_remaining": st.cooldown_remaining,
            "last_guard_reason": d.reason,
            "top_call_score": d.top_call_score,
            "top_put_score": d.top_put_score,
            "direction_score_edge": d.score_edge,
        })
        return base


def create_v14_5_options_runner(
    symbol: str,
    trading_client: Any,
    option_data_client: Any,
    paper_mode: bool = True,
    log_dir: str = "HFT/logs/v14_5_options",
) -> V14_5_StableOptionsRunner:
    return V14_5_StableOptionsRunner(
        symbol=symbol,
        trading_client=trading_client,
        option_data_client=option_data_client,
        config=get_v14_5_config(),
        paper_mode=paper_mode,
        log_dir=str(Path(log_dir) / symbol.upper()),
    )
