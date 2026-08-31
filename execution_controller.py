"""
Fresh-market wrapper for the V14_2_1 execution controller.

The original controller implementation is preserved verbatim in
``execution_controller_legacy.py``.  This module re-exports its public API and
wraps only ``run_controller_loop`` so the live Alpaca signal engine cannot
reuse the one-time bootstrap snapshot forever.

Freshness invariants
--------------------
1. If a signal engine exposes ``poll_latest()``, market data is refreshed before
   the controller reads market state or generates candidates.
2. A failed refresh fails closed: the legacy controller's outer exception
   handler aborts that cycle before new candidate orders can be created.
3. Signals are generated only for symbols whose 1-minute bar timestamp advanced
   during the refresh.  A 45-second controller loop therefore cannot trade the
   same completed 1-minute bar twice.
4. Engines without ``poll_latest()`` (tests/replay/mocks) retain the legacy
   behavior unchanged.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Set

import execution_controller_legacy as _legacy
from execution_controller_legacy import *  # noqa: F401,F403 - preserve public API


class FreshMarketSignalEngine:
    """Thin freshness guard around a live signal engine.

    The controller asks the signal engine for costs/market state before it asks
    for new signals.  Refreshing at these access points keeps loss governance,
    hold scoring, and candidate generation on the same newest bar without
    changing the legacy controller's order/reconciliation logic.
    """

    def __init__(self, engine: Any, refresh_interval_seconds: float = 1.0):
        self._engine = engine
        self._refresh_interval_seconds = max(0.0, float(refresh_interval_seconds))
        self._last_refresh_monotonic = float("-inf")
        self._fresh_symbols: Optional[Set[str]] = None

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

        # Intentionally do not swallow exceptions.  The legacy controller wraps
        # each cycle in an exception boundary; propagating here makes a data-feed
        # failure fail closed instead of evaluating stale bars.
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
            # Generic engines without per-symbol buffers still benefit from the
            # refresh call; do not impose symbol-level filtering on them.
            self._fresh_symbols = None

        self._last_refresh_monotonic = time.monotonic()

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
            return self._engine.generate_signals()

        # No completed bar advanced since the prior cycle.  Returning no
        # candidates is deliberate: broker reconciliation still runs, but no
        # stale entry/exit signal is regenerated from the same bar.
        if not self._fresh_symbols:
            return []

        # AlpacaTrader.generate_signals() iterates self.sym_states.  Temporarily
        # narrow that mapping to only symbols with a new bar so stale symbols do
        # not mutate signal history/cooldowns or emit repeated TradeSignals.
        all_states = states
        fresh_states = {
            symbol: state
            for symbol, state in all_states.items()
            if symbol in self._fresh_symbols
        }
        self._engine.sym_states = fresh_states
        try:
            return self._engine.generate_signals()
        finally:
            self._engine.sym_states = all_states
            # Prevent a second generate_signals() call from reusing this same
            # refresh window.  The next controller cycle must poll again.
            self._fresh_symbols = set()


def run_controller_loop(signal_engine, broker_adapter, campaign_book, policy,
                        eastern_tz, check_interval: int = 45):
    """Run the legacy controller with live market-data freshness enforced."""
    guarded_engine = signal_engine
    if callable(getattr(signal_engine, "poll_latest", None)):
        # The legacy loop sleeps check_interval at the end of each normal cycle.
        # A slightly shorter guard interval guarantees the next cycle refreshes
        # while avoiding repeated REST polls within a normal single cycle.
        refresh_interval = max(1.0, float(check_interval) * 0.80)
        guarded_engine = FreshMarketSignalEngine(
            signal_engine,
            refresh_interval_seconds=refresh_interval,
        )
        print(
            f"  Fresh market-data guard: ENABLED "
            f"(refresh <= every {refresh_interval:.1f}s; stale-bar signals blocked)"
        )

    return _legacy.run_controller_loop(
        signal_engine=guarded_engine,
        broker_adapter=broker_adapter,
        campaign_book=campaign_book,
        policy=policy,
        eastern_tz=eastern_tz,
        check_interval=check_interval,
    )


# Keep the historical alias pointed at the guarded implementation.
run_execution_loop = run_controller_loop


if __name__ == "__main__":
    # Preserve the old ``python execution_controller.py`` regression-test entry.
    _legacy._run_regression_tests()
