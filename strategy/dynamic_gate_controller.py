"""
Dynamic Gate Controller for V14.2 Core Runner
================================================

Replaces two problems in the static gate design:

1. Arbitrary fixed trade-count caps (max_trades_per_day=2, etc.) that throttle
   the bot independent of whether capital is actually available or whether
   the setups are good. RiskManager now derives trade cadence from capital
   headroom instead (see risk_manager.py) — this controller focuses on
   quality/opportunity, not capital, and never touches the exposure cap or
   kill switches.

2. Permanent lockout: RouteExpectancyMonitor/EnvironmentExpectancyMonitor
   flip a route or environment to DISABLED/BLOCKED based on rolling PF, but
   entry gating (route_monitor.is_route_allowed) is checked *before* a new
   ticket/trade is created — so a disabled route never gets new trades to
   re-evaluate itself with, and stays disabled forever. This controller
   replaces the permanent block with a cooldown, followed by a small number
   of size-reduced "probation" trades so the route can re-earn trust with
   fresh data (or confirm it should stay off).

Everything here only ever adjusts *entry-quality* thresholds (how selective
we are about which setups we take) and *position size* (how much we risk on
a given entry). It never raises the hard capital limits (max_open_debit_
exposure_pct, daily/weekly kill switches) — those stay fixed as the actual
"don't blow the account" backstop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG

# Config keys whose *value* gets scaled by a route's confidence multiplier.
# `higher_is_stricter=True` means increasing the value makes entry harder
# (a min-threshold), so a proven route's multiplier (<1.0) lowers the bar;
# `higher_is_stricter=False` means increasing the value makes entry *easier*
# (a max-threshold, e.g. max adverse move allowed), so a proven route's
# multiplier should raise it — we invert the multiplier for those keys.
_MIN_THRESHOLD_KEYS = (
    "watch_min_route_score",
    "watch_min_ic_spread",
    "watch_min_ev_over_debit",
    "watch_min_option_liquidity",
    "confirm_min_directional_atr",
    "confirm_min_mfe_velocity",
    "package_min_pwin",
    "package_min_q",
)
_MAX_THRESHOLD_KEYS = (
    "confirm_max_adverse_atr",
)


@dataclass
class RouteAdjustment:
    """Confidence-derived adjustment for a route."""
    route: str
    multiplier: float          # 1.0 = neutral, <1.0 = loosened, >1.0 = tightened
    state: str                 # FAVORED, NORMAL, CAUTION, PROBATION
    size_multiplier: float = 1.0
    reason: str = ""


class DynamicGateController:
    """
    Sits between the core runner and (RouteExpectancyMonitor,
    EnvironmentExpectancyMonitor) to turn their binary ACTIVE/DISABLED and
    ALLOWED/BLOCKED signals into continuous, recoverable ones.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        route_monitor=None,
        env_monitor=None,
    ):
        self.cfg = config or CFG
        self.route_monitor = route_monitor
        self.env_monitor = env_monitor

        self._mult_min = self.cfg.get("gate_multiplier_min", 0.85)
        self._mult_max = self.cfg.get("gate_multiplier_max", 1.25)

        # Cooldown/probation bookkeeping (per route / per environment label).
        self._route_disabled_since: Dict[str, datetime] = {}
        self._route_probation_used: Dict[str, int] = {}
        self._env_blocked_since: Dict[str, datetime] = {}
        self._env_probation_used: Dict[str, int] = {}

    # ─── Route confidence → threshold + size adjustment ───────────────────

    def get_route_adjustment(self, route: str) -> RouteAdjustment:
        """
        Bounded confidence multiplier for a route's entry thresholds, based
        on its rolling profit factor. Insufficient sample size -> neutral.
        """
        if self.route_monitor is None:
            return RouteAdjustment(route, 1.0, "NORMAL", 1.0, "no_monitor")

        trades = self.route_monitor.get_trade_count(route, 50)
        if trades < 10:
            return RouteAdjustment(route, 1.0, "NORMAL", 1.0, "insufficient_data")

        pf50 = self.route_monitor.get_profit_factor(route, 50)

        if pf50 >= 2.0:
            mult, state, size = 0.85, "FAVORED", 1.0
        elif pf50 >= 1.5:
            mult, state, size = 0.93, "NORMAL", 1.0
        elif pf50 >= 1.0:
            mult, state, size = 1.10, "CAUTION", 0.75
        else:
            mult, state, size = 1.25, "CAUTION", 0.5

        mult = max(self._mult_min, min(self._mult_max, mult))
        return RouteAdjustment(route, mult, state, size, f"pf50={pf50:.2f} n={trades}")

    def get_effective_thresholds(self, route: str) -> Dict[str, float]:
        """
        Return a dict of {config_key: adjusted_value} for the subset of
        opportunity-selection thresholds that adapt per route. Callers merge
        this over the base config for that one evaluation — spread-quality /
        execution-risk keys are deliberately excluded and always stay fixed.
        """
        adj = self.get_route_adjustment(route)
        out: Dict[str, float] = {}
        for key in _MIN_THRESHOLD_KEYS:
            if key in self.cfg:
                out[key] = self.cfg[key] * adj.multiplier
        for key in _MAX_THRESHOLD_KEYS:
            if key in self.cfg:
                # Inverted: a loosening multiplier (<1.0) should *raise* a
                # max-allowed threshold, not lower it.
                inverse = 1.0 / adj.multiplier if adj.multiplier > 0 else 1.0
                out[key] = self.cfg[key] * inverse
        return out

    # ─── Route cooldown → probation recovery ───────────────────────────────

    def route_trade_allowed(self, route: str) -> Tuple[bool, str, float]:
        """
        Returns (allowed, state, size_multiplier).

        ACTIVE / SOFT_SIZE_ONLY pass through with the monitor's own sizing.
        DISABLED no longer means "forever": after a cooldown window it opens
        a limited number of size-reduced probation trades so the route can
        generate fresh data and requalify (or confirm it should stay off).
        """
        if self.route_monitor is None:
            return True, "ACTIVE", 1.0

        status = self.route_monitor.get_route_status(route)

        if status == "ACTIVE":
            self._route_disabled_since.pop(route, None)
            self._route_probation_used.pop(route, None)
            return True, "ACTIVE", 1.0

        if status == "SOFT_SIZE_ONLY":
            self._route_disabled_since.pop(route, None)
            self._route_probation_used.pop(route, None)
            return True, "SOFT_SIZE_ONLY", 0.5

        # status == "DISABLED"
        since = self._route_disabled_since.setdefault(route, datetime.utcnow())
        cooldown = timedelta(hours=self.cfg.get("route_probation_cooldown_hours", 24))
        if datetime.utcnow() - since < cooldown:
            return False, "COOLDOWN", 0.0

        used = self._route_probation_used.get(route, 0)
        limit = self.cfg.get("route_probation_trade_limit", 1)
        if used < limit:
            return True, "PROBATION", self.cfg.get("route_probation_size_multiplier", 0.5)

        return False, "PROBATION_EXHAUSTED", 0.0

    def record_route_probation_trade(self, route: str) -> None:
        """Call after a probation-state trade is actually placed."""
        self._route_probation_used[route] = self._route_probation_used.get(route, 0) + 1

    # ─── Environment cooldown → probation recovery ─────────────────────────

    def environment_trade_allowed(self, environment) -> Tuple[bool, str, float]:
        """Same treatment as routes, for EnvironmentExpectancyMonitor labels."""
        if self.env_monitor is None:
            return True, "ALLOWED", 1.0

        label = environment.label
        status = self.env_monitor.get_environment_status(environment)

        if status == "ALLOWED":
            self._env_blocked_since.pop(label, None)
            self._env_probation_used.pop(label, None)
            return True, "ALLOWED", 1.0

        # status == "BLOCKED"
        since = self._env_blocked_since.setdefault(label, datetime.utcnow())
        cooldown = timedelta(hours=self.cfg.get("env_probation_cooldown_hours", 12))
        if datetime.utcnow() - since < cooldown:
            return False, "COOLDOWN", 0.0

        used = self._env_probation_used.get(label, 0)
        limit = self.cfg.get("env_probation_trade_limit", 1)
        if used < limit:
            return True, "PROBATION", self.cfg.get("env_probation_size_multiplier", 0.5)

        return False, "PROBATION_EXHAUSTED", 0.0

    def record_env_probation_trade(self, environment) -> None:
        label = environment.label
        self._env_probation_used[label] = self._env_probation_used.get(label, 0) + 1
