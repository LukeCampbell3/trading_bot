"""
Fresh-market and profit-reversal wrapper for the V14_2_1 execution controller.

The original controller remains preserved in ``execution_controller_legacy.py``.
This module re-exports its public API, then strengthens the live controller path
with four execution invariants:

1. Fresh bars only: stale 1-minute bars cannot regenerate signals.
2. Profit ratchet: once profit protection is armed, peak P/L creates a rising
   floor; a sufficiently large giveback exits the remaining position.
3. Reduction recovery: a completed partial profit trim returns to
   ``PROTECT_PROFIT`` instead of getting stranded in ``REDUCING``.
4. Exit-first safety: safety modes that block new entries still allow fresh
   flatten/reversal signals to reduce exposure.

The wrapper intentionally leaves the predictive model untouched.  These changes
repair execution semantics around existing signals and broker state.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import execution_controller_legacy as _legacy
from execution_controller_legacy import *  # noqa: F401,F403 - preserve public API


# ---------------------------------------------------------------------------
# Profit-ratchet defaults
# ---------------------------------------------------------------------------
# These are execution defaults, not model thresholds.  They are intentionally
# configurable through same-named attributes on ExecutionPolicy when a caller
# wants to tune them without changing this module.
PROFIT_TRAIL_MIN_GIVEBACK_PCT = 0.0035
PROFIT_TRAIL_MAX_GIVEBACK_PCT = 0.0100
PROFIT_TRAIL_GIVEBACK_FRACTION = 0.35
PROFIT_TRIM_FRACTION = 0.50


def _policy_float(policy: Any, name: str, default: float) -> float:
    try:
        return float(getattr(policy, name))
    except (AttributeError, TypeError, ValueError):
        return float(default)


def _ensure_profit_tracking(campaign: Any) -> None:
    """Attach backward-compatible profit-tracking fields to legacy campaigns."""
    if not hasattr(campaign, "_profit_trimmed"):
        campaign._profit_trimmed = False
    if not hasattr(campaign, "_profit_trail_armed"):
        campaign._profit_trail_armed = False
    if not hasattr(campaign, "_profit_floor_pct"):
        campaign._profit_floor_pct = 0.0


def _clear_profit_tracking(campaign: Any) -> None:
    """Reset wrapper-owned state after a position is fully closed."""
    campaign._profit_trimmed = False
    campaign._profit_trail_armed = False
    campaign._profit_floor_pct = 0.0
    if hasattr(campaign, "best_pnl_pct"):
        campaign.best_pnl_pct = 0.0
    if hasattr(campaign, "current_pnl_pct"):
        campaign.current_pnl_pct = 0.0


class ProfitAwareLossGovernor(_legacy.LossGovernor):
    """Legacy loss governor plus one-time trimming and peak-profit ratcheting."""

    def _trail_floor(self, campaign: Any) -> float:
        _ensure_profit_tracking(campaign)
        peak = max(0.0, float(getattr(campaign, "best_pnl_pct", 0.0) or 0.0))
        min_giveback = _policy_float(
            self.policy, "profit_trail_min_giveback_pct",
            PROFIT_TRAIL_MIN_GIVEBACK_PCT,
        )
        max_giveback = _policy_float(
            self.policy, "profit_trail_max_giveback_pct",
            PROFIT_TRAIL_MAX_GIVEBACK_PCT,
        )
        giveback_fraction = _policy_float(
            self.policy, "profit_trail_giveback_fraction",
            PROFIT_TRAIL_GIVEBACK_FRACTION,
        )

        # Let larger winners breathe more in absolute terms, but never allow
        # the protected floor to move downward.
        giveback = max(min_giveback, min(max_giveback, peak * giveback_fraction))
        floor = max(float(_legacy.PROFIT_NO_ADDING_PCT), peak - giveback)
        campaign._profit_floor_pct = max(
            float(getattr(campaign, "_profit_floor_pct", 0.0) or 0.0),
            floor,
        )
        return campaign._profit_floor_pct

    def evaluate_profit_protection(
        self,
        campaign: Any,
        current_price: float,
    ) -> Optional[str]:
        """
        Return one of:
          None, no_adding, protect_profit, trim_trail, trail_exit

        Unlike the legacy implementation, this records peak P/L and converts
        that peak into a ratcheting floor after +1.25% profit is reached.
        """
        if campaign.entry_price is None or campaign.entry_price <= 0:
            return None
        if current_price <= 0:
            return None

        _ensure_profit_tracking(campaign)

        pnl_pct = (current_price - campaign.entry_price) / campaign.entry_price
        campaign.current_pnl_pct = pnl_pct
        campaign.best_pnl_pct = max(
            float(getattr(campaign, "best_pnl_pct", 0.0) or 0.0),
            pnl_pct,
        )

        if campaign.best_pnl_pct >= _legacy.PROFIT_PROTECT_PCT:
            campaign._profit_trail_armed = True
            floor = self._trail_floor(campaign)
            if pnl_pct <= floor:
                return "trail_exit"

        if (
            pnl_pct >= _legacy.PROFIT_TRIM_TRAIL_PCT
            and not campaign._profit_trimmed
        ):
            return "trim_trail"
        if pnl_pct >= _legacy.PROFIT_PROTECT_PCT:
            return "protect_profit"
        if pnl_pct >= _legacy.PROFIT_NO_ADDING_PCT:
            return "no_adding"
        return None

    def apply_profit_action(self, campaign: Any, action: str) -> str:
        _ensure_profit_tracking(campaign)

        if action == "trail_exit":
            campaign.target_qty = 0
            campaign.blocked_setup_fingerprint = campaign.setup_fingerprint
            campaign.last_exit_time = datetime.utcnow()
            campaign.last_exit_reason = "profit_trail"
            if campaign.state != "LOCKED_ERROR":
                campaign.state = "EXITING"
            return campaign.state

        if action == "trim_trail":
            if campaign._profit_trimmed:
                return campaign.state

            trim_fraction = _policy_float(
                self.policy, "profit_trim_fraction", PROFIT_TRIM_FRACTION
            )
            trim_fraction = min(0.95, max(0.05, trim_fraction))
            keep_fraction = 1.0 - trim_fraction
            campaign.target_qty = round(
                max(0.01, float(campaign.target_qty) * keep_fraction), 2
            )
            campaign._profit_trimmed = True
            if campaign.state != "REDUCING":
                campaign.transition_to("REDUCING")
            return campaign.state

        if action == "protect_profit":
            campaign._profit_trail_armed = True
            if campaign.state in ("ACTIVE", "BUILDING"):
                campaign.transition_to("PROTECT_PROFIT")
            return campaign.state

        if action == "no_adding":
            return campaign.state

        return campaign.state


class ProfitAwareExecutionReconciler(_legacy.ExecutionReconciler):
    """Reconciler that cannot strand a filled profit trim in REDUCING."""

    def __init__(self, broker: Any, policy: Any, eastern_tz: Any):
        super().__init__(broker, policy, eastern_tz)
        self.allow_increases = True

    def _increase_exposure(
        self,
        symbol: str,
        campaign: Any,
        broker_state: Any,
        desired_qty: float,
    ) -> str:
        if not self.allow_increases:
            return "entry_blocked(safety_no_new_entries)"
        return super()._increase_exposure(
            symbol, campaign, broker_state, desired_qty
        )

    def reconcile_symbol(
        self,
        symbol: str,
        campaign: Any,
        broker_state: Any,
    ) -> str:
        desired = campaign.target_qty
        effective = broker_state.effective_exposure

        # Legacy code treats desired==effective as "in_sync" but has no state
        # transition for REDUCING.  Once a partial profit sale fills, put the
        # surviving position back under profit governance immediately.  Do not
        # treat a still-open sell order as a completed reduction merely because
        # pending sells make effective exposure equal the target.
        if (
            campaign.state == "REDUCING"
            and desired == effective
            and broker_state.position_qty > 0
            and not broker_state.open_sell_order_ids
            and float(broker_state.position_qty) <= float(desired) + 1e-9
        ):
            campaign.state = "PROTECT_PROFIT"
            self._log(
                symbol,
                "REDUCTION_FILLED",
                f"position={broker_state.position_qty} target={desired}; "
                "returned to PROTECT_PROFIT",
            )
            return "reduction_complete"

        result = super().reconcile_symbol(symbol, campaign, broker_state)

        if (
            broker_state.position_qty == 0
            and campaign.state in ("FLAT", "COOLDOWN_BLOCKED")
        ):
            _clear_profit_tracking(campaign)

        return result


# Export strengthened classes for callers/tests that import from this wrapper.
LossGovernor = ProfitAwareLossGovernor
ExecutionReconciler = ProfitAwareExecutionReconciler


class FreshMarketSignalEngine:
    """
    Freshness guard plus a conservative signal-deterioration exit guard.

    The deterioration guard only activates when the underlying signal engine
    already exposes its historical v14.2 exit knobs:
      * ``base_exit_threshold``
      * ``signal_reversal_delta``

    A still-positive long signal is converted to ``flat`` only when it has
    fallen by at least ``signal_reversal_delta`` and has also weakened below the
    engine's own ``base_exit_threshold``.  This wires previously-unused exit
    intent into the active controller without changing entry behavior.
    """

    def __init__(self, engine: Any, refresh_interval_seconds: float = 1.0):
        self._engine = engine
        self._refresh_interval_seconds = max(
            0.0, float(refresh_interval_seconds)
        )
        self._last_refresh_monotonic = float("-inf")
        self._fresh_symbols: Optional[Set[str]] = None
        self._last_expected_return_bps: Dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def _snapshot_bar_timestamps(self) -> Dict[str, Any]:
        states = getattr(self._engine, "sym_states", None)
        if not isinstance(states, dict):
            return {}

        out: Dict[str, Any] = {}
        for symbol, state in states.items():
            buf = getattr(state, "buf", None)
            out[symbol] = getattr(buf, "last_timestamp", None)
        return out

    def _ensure_refreshed(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh_monotonic < self._refresh_interval_seconds:
            return

        poll_latest = getattr(self._engine, "poll_latest", None)
        if not callable(poll_latest):
            return

        before = self._snapshot_bar_timestamps()

        # Fail closed on feed errors; the controller's outer exception boundary
        # will abort the cycle instead of acting on stale bars.
        poll_latest()

        after = self._snapshot_bar_timestamps()
        if after:
            fresh: Set[str] = set()
            for symbol, new_ts in after.items():
                old_ts = before.get(symbol)
                if new_ts is not None and (old_ts is None or new_ts > old_ts):
                    fresh.add(symbol)
            self._fresh_symbols = fresh
        else:
            self._fresh_symbols = None

        self._last_refresh_monotonic = time.monotonic()

    def _apply_reversal_guard(self, signals: Iterable[Any]) -> List[Any]:
        signals = list(signals)

        exit_threshold = getattr(self._engine, "base_exit_threshold", None)
        reversal_delta = getattr(self._engine, "signal_reversal_delta", None)
        try:
            weak_bps = float(exit_threshold) * 10000.0
            reversal_bps = float(reversal_delta) * 10000.0
        except (TypeError, ValueError):
            weak_bps = None
            reversal_bps = None

        for signal in signals:
            symbol = getattr(signal, "symbol", "")
            try:
                current_bps = float(signal.expected_return_bps)
            except (AttributeError, TypeError, ValueError):
                continue

            previous_bps = self._last_expected_return_bps.get(symbol)

            if (
                previous_bps is not None
                and weak_bps is not None
                and reversal_bps is not None
                and reversal_bps > 0
                and getattr(signal, "direction", None) == "long"
                and 0 < current_bps <= weak_bps
                and (previous_bps - current_bps) >= reversal_bps
            ):
                signal.direction = "flat"
                base_reason = getattr(signal, "reason", "") or ""
                guard_reason = (
                    f"reversal_guard(drop={previous_bps-current_bps:.2f}bps,"
                    f"weak={current_bps:.2f}bps)"
                )
                signal.reason = (
                    f"{base_reason}|{guard_reason}"
                    if base_reason else guard_reason
                )

            self._last_expected_return_bps[symbol] = current_bps

        return signals

    def get_market_state(self, symbol: str):
        self._ensure_refreshed()
        return self._engine.get_market_state(symbol)

    def estimate_costs(self, symbol: str):
        self._ensure_refreshed()
        return self._engine.estimate_costs(symbol)

    def generate_signals(self):
        self._ensure_refreshed()

        states = getattr(self._engine, "sym_states", None)
        if not isinstance(states, dict) or self._fresh_symbols is None:
            return self._apply_reversal_guard(
                self._engine.generate_signals()
            )

        if not self._fresh_symbols:
            return []

        all_states = states
        fresh_states = {
            symbol: state
            for symbol, state in all_states.items()
            if symbol in self._fresh_symbols
        }
        self._engine.sym_states = fresh_states
        try:
            signals = self._engine.generate_signals()
            return self._apply_reversal_guard(signals)
        finally:
            self._engine.sym_states = all_states
            self._fresh_symbols = set()


def _decision_allowed_by_safety(decision: Any, safety_manager: Any) -> bool:
    """
    Safety modes may block *increases* without suppressing flatten signals.

    ORDER_RECONCILIATION_ONLY remains strict because broker contradictions must
    be settled before strategy-driven orders resume.
    """
    mode = getattr(safety_manager, "mode", None)
    if mode == _legacy.SafetyMode.ORDER_RECONCILIATION_ONLY:
        return decision.action == "NO_ACTION"

    if decision.action == "TARGET_ZERO":
        return bool(safety_manager.allows_exits())
    if decision.action == "SET_TARGET":
        return bool(safety_manager.allows_new_entries())
    return True


def _managed_position_state(campaign: Any) -> bool:
    return campaign.state in (
        "ACTIVE",
        "BUILDING",
        "PROTECT_PROFIT",
        "REVALIDATE_LOSER",
        "REDUCING",
    )


def run_controller_loop(
    signal_engine,
    broker_adapter,
    campaign_book,
    policy,
    eastern_tz,
    check_interval: int = 45,
):
    """Run the V14_2_1 controller with profit-ratchet and exit-first fixes."""

    guarded_engine = signal_engine
    if callable(getattr(signal_engine, "poll_latest", None)):
        refresh_interval = max(1.0, float(check_interval) * 0.80)
        guarded_engine = FreshMarketSignalEngine(
            signal_engine,
            refresh_interval_seconds=refresh_interval,
        )
        print(
            f"  Fresh market-data guard: ENABLED "
            f"(refresh <= every {refresh_interval:.1f}s; stale-bar signals blocked)"
        )

    reconciler = ProfitAwareExecutionReconciler(
        broker_adapter, policy, eastern_tz
    )
    loss_governor = ProfitAwareLossGovernor(policy)
    safety_manager = _legacy.SafetyModeManager()
    hold_scorer = _legacy.HoldScorer()
    admission = _legacy.PortfolioAdmission(policy)
    campaign_book.policy_ref = policy

    trades_today = 0
    day_start_equity = None
    current_day = None

    print(f"\n{'=' * 70}")
    print(
        f"  Execution Controller {_legacy.VERSION} "
        "- Profit-Ratchet / Exit-First"
    )
    print(f"  Symbols: {len(campaign_book.campaigns)}")
    print(
        f"  Policy: entries "
        f"{policy.no_new_entries_before}-{policy.no_new_entries_after} ET"
    )
    print(f"  Max positions: {policy.max_concurrent_positions}")
    print(
        "  Profit protection: "
        f"arm={_legacy.PROFIT_PROTECT_PCT*100:.2f}% "
        f"trim={_legacy.PROFIT_TRIM_TRAIL_PCT*100:.2f}% "
        f"min-floor={_legacy.PROFIT_NO_ADDING_PCT*100:.2f}%"
    )
    print(f"{'=' * 70}\n")

    while True:
        try:
            now_et = datetime.now(eastern_tz)
            now_utc = datetime.utcnow()

            today = now_et.date()
            if current_day != today:
                current_day = today
                trades_today = 0
                admission.reset_day(today)
                loss_governor.session_locked_symbols.clear()
                snapshot = broker_adapter.fetch_snapshot()
                day_start_equity = snapshot.account_equity
                print(
                    f"\n  -- New day: {today} | "
                    f"equity=${day_start_equity:,.2f}"
                )

            if now_et.weekday() >= 5:
                time.sleep(300)
                continue

            # 1. Broker truth / 2. rebuild ledgers
            snapshot = broker_adapter.fetch_snapshot()
            campaign_book.sync_from_broker(snapshot)
            for _, campaign in campaign_book.items():
                if campaign.state == "FLAT":
                    _clear_profit_tracking(campaign)

            safety_manager.evaluate_portfolio_health(
                snapshot,
                campaign_book.campaigns,
                day_start_equity or snapshot.account_equity,
            )

            if safety_manager.mode == _legacy.SafetyMode.MANUAL_REVIEW_REQUIRED:
                print(
                    f"  [{now_et:%H:%M:%S}] MANUAL REVIEW REQUIRED - halted"
                )
                time.sleep(300)
                continue

            # 3. contradictions / 4. stale order cleanup
            reconciler.resolve_all_contradictions(
                campaign_book.campaigns, snapshot
            )
            reconciler.cancel_stale_orders(snapshot, now_utc)

            if _legacy.should_cancel_premarket_orders(now_et, policy):
                for order in snapshot.all_open_orders:
                    if _legacy.is_bot_order(
                        order.get("client_order_id", "")
                    ):
                        broker_adapter.cancel_order(order["id"])
                time.sleep(check_interval)
                continue

            # 5. Refresh broker truth
            snapshot = broker_adapter.fetch_snapshot()

            # 6. Rescore positions using the actual current market price.
            for symbol, campaign in campaign_book.items():
                if not _managed_position_state(campaign):
                    continue
                bs = snapshot.get_symbol_state(symbol)
                ms = guarded_engine.get_market_state(symbol)
                current_price = (
                    ms.last_price
                    if ms.last_price > 0
                    else (bs.avg_entry_price or 0.0)
                )
                costs = guarded_engine.estimate_costs(symbol)
                hold_scorer.score(
                    campaign,
                    current_price=current_price,
                    spread_bps=costs.spread_bps,
                )
                campaign.bars_held += 1

            # 7. Loss governance + one-time trim + peak-profit ratchet.
            for symbol, campaign in campaign_book.items():
                if not _managed_position_state(campaign):
                    continue

                bs = snapshot.get_symbol_state(symbol)
                ms = guarded_engine.get_market_state(symbol)
                current_price = (
                    ms.last_price
                    if ms.last_price > 0
                    else (bs.avg_entry_price or 0.0)
                )

                loss_action = loss_governor.evaluate(
                    campaign, current_price
                )
                if loss_action:
                    has_fresh = (
                        campaign.last_signal_time
                        and (
                            now_utc - campaign.last_signal_time
                        ).total_seconds() < 120
                    )
                    new_state = loss_governor.apply_loss_action(
                        campaign,
                        loss_action,
                        has_fresh_confirm=bool(has_fresh),
                    )
                    if new_state in ("EXITING", "LOCKED_ERROR"):
                        continue

                profit_action = loss_governor.evaluate_profit_protection(
                    campaign, current_price
                )
                if profit_action:
                    loss_governor.apply_profit_action(
                        campaign, profit_action
                    )

            # 8. Reduce / exit immediately.
            for symbol, campaign in campaign_book.items():
                if campaign.state in ("EXITING", "REDUCING"):
                    bs = snapshot.get_symbol_state(symbol)
                    reconciler.reconcile_symbol(symbol, campaign, bs)
                elif (
                    campaign.state == "LOCKED_ERROR"
                    and campaign.target_qty == 0
                ):
                    bs = snapshot.get_symbol_state(symbol)
                    if (
                        bs.position_qty > 0
                        or bs.open_buy_order_ids
                    ):
                        reconciler.reconcile_symbol(
                            symbol, campaign, bs
                        )

            if safety_manager.is_liquidate_only():
                for symbol, campaign in campaign_book.items():
                    if _managed_position_state(campaign):
                        campaign_book.set_target_zero(
                            symbol, "liquidate_mode"
                        )
                liquidate_snapshot = broker_adapter.fetch_snapshot()
                reconciler.allow_increases = False
                reconciler.reconcile_all(
                    campaign_book.campaigns, liquidate_snapshot
                )
                time.sleep(check_interval)
                continue

            # 9. Refresh / 10. contradiction re-check.
            snapshot = broker_adapter.fetch_snapshot()
            unresolved = reconciler.resolve_all_contradictions(
                campaign_book.campaigns, snapshot
            )
            if unresolved > 0:
                safety_manager.set_mode(
                    _legacy.SafetyMode.ORDER_RECONCILIATION_ONLY,
                    f"unresolved_contradictions={unresolved}",
                )

            # Strict reconciliation-only mode stops strategy orders, but other
            # safety modes continue to evaluate fresh exit signals.
            if (
                safety_manager.mode
                == _legacy.SafetyMode.ORDER_RECONCILIATION_ONLY
            ):
                reconciler.allow_increases = False
                time.sleep(check_interval)
                continue

            # 11. Generate candidates even when new entries are disabled so
            # negative/reversal signals can still flatten held positions.
            signals = guarded_engine.generate_signals()
            in_window, window_reason = _legacy.is_in_trading_window(
                now_et, policy
            )

            # 12. Evaluate/admit.
            admission.reset_cycle()
            admitted_signals: List[Tuple[Any, Any]] = []

            for signal in signals:
                campaign = campaign_book.get(signal.symbol)
                broker_state = snapshot.get_symbol_state(signal.symbol)

                # Never let a long signal resurrect a position while an exit or
                # partial reduction is still being reconciled.
                if (
                    signal.direction != "flat"
                    and campaign.state in (
                        "EXITING",
                        "REDUCING",
                        "LOCKED_ERROR",
                        "COOLDOWN_BLOCKED",
                    )
                ):
                    continue

                market_state = guarded_engine.get_market_state(
                    signal.symbol
                )
                costs = guarded_engine.estimate_costs(signal.symbol)

                decision = _legacy.evaluate_signal(
                    signal=signal,
                    campaign=campaign,
                    broker_state=broker_state,
                    market_state=market_state,
                    costs=costs,
                    account_equity=snapshot.account_equity,
                    price=market_state.last_price,
                    policy=policy,
                    buying_power=snapshot.account_buying_power,
                    loss_governor=loss_governor,
                )

                if not _decision_allowed_by_safety(
                    decision, safety_manager
                ):
                    continue

                if (
                    decision.action == "SET_TARGET"
                    and not in_window
                ):
                    decision = _legacy.Decision(
                        action="NO_ACTION",
                        reason=window_reason,
                    )

                if decision.action == "TARGET_ZERO":
                    campaign_book.set_target_zero(
                        signal.symbol, decision.reason
                    )
                    continue

                if decision.action != "SET_TARGET":
                    continue

                can_admit, _ = admission.can_admit(
                    signal,
                    decision.target_qty,
                    campaign_book.campaigns,
                    snapshot,
                )
                if not can_admit:
                    continue

                admitted_signals.append((signal, decision))

            # 13. Assign targets.
            for signal, decision in admitted_signals:
                campaign_book.set_target(
                    signal.symbol,
                    signal,
                    decision.target_qty,
                    signal.invalidation_price,
                )
                admission.record_admission()
                trades_today += 1

            # 14. Reconcile.  Safety state also gates pending target increases,
            # not just newly-created signals.
            fresh_snapshot = broker_adapter.fetch_snapshot()
            reconciler.allow_increases = (
                safety_manager.allows_new_entries()
            )
            results = reconciler.reconcile_all(
                campaign_book.campaigns, fresh_snapshot
            )

            # 15. Audit.
            active = sum(
                1
                for result in results.values()
                if result not in ("in_sync", "no_action")
            )
            n_pos = fresh_snapshot.n_active_positions
            if (
                active > 0
                or int(time.time()) % 120 < check_interval
            ):
                exp_pct = (
                    fresh_snapshot.total_exposure_notional
                    / max(fresh_snapshot.account_equity, 1.0)
                )
                print(
                    f"  [{now_et:%H:%M:%S}] pos={n_pos} "
                    f"exposure={exp_pct:.1%} trades={trades_today} "
                    f"safety={safety_manager.mode.value} "
                    f"reconciled={active}"
                )

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print(
                f"\n  Shutting down {_legacy.VERSION} "
                "profit-ratchet controller..."
            )
            break
        except Exception as exc:
            print(f"  Loop error: {exc}")
            import traceback
            traceback.print_exc()
            time.sleep(check_interval)


# Historical alias points at the strengthened implementation.
run_execution_loop = run_controller_loop


if __name__ == "__main__":
    _legacy._run_regression_tests()
