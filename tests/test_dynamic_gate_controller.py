"""
Tests for the Dynamic Gate Controller: adaptive per-route thresholds and
route/environment cooldown -> probation recovery (replacing permanent
lockout).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

import pytest

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG
from strategy.dynamic_gate_controller import DynamicGateController
from telemetry.route_expectancy_monitor import RouteExpectancyMonitor
from telemetry.environment_expectancy_monitor import (
    EnvironmentExpectancyMonitor, EnvironmentState,
)


def make_env(label_suffix=""):
    return EnvironmentState(
        timestamp=datetime.utcnow(),
        vix_level="NORMAL" + label_suffix,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE EXPECTANCY MONITOR — accessor
# ═══════════════════════════════════════════════════════════════════════════

class TestRouteExpectancyAccessors:
    def test_get_trade_count(self, tmp_path):
        mon = RouteExpectancyMonitor(log_dir=str(tmp_path))
        assert mon.get_trade_count("VWAP_PULLBACK") == 0
        mon.record_trade("VWAP_PULLBACK", 50.0)
        mon.record_trade("VWAP_PULLBACK", -20.0)
        assert mon.get_trade_count("VWAP_PULLBACK") == 2
        assert mon.get_trade_count("OTHER_ROUTE") == 0


class TestEnvironmentExpectancyAccessors:
    def test_get_environment_status_defaults_allowed(self, tmp_path):
        mon = EnvironmentExpectancyMonitor(log_dir=str(tmp_path))
        env = make_env()
        assert mon.get_environment_status(env) == "ALLOWED"

    def test_get_environment_status_blocks_on_negative_expectancy(self, tmp_path):
        mon = EnvironmentExpectancyMonitor(log_dir=str(tmp_path))
        env = make_env()
        for _ in range(10):
            mon.record_trade(env, "AAPL", -10.0)
        assert mon.get_environment_status(env) == "BLOCKED"


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

class TestRouteAdjustment:
    def setup_method(self):
        self.route_monitor = RouteExpectancyMonitor(log_dir="HFT/logs/test_dgc")
        self.env_monitor = EnvironmentExpectancyMonitor(log_dir="HFT/logs/test_dgc")
        self.controller = DynamicGateController(
            config=dict(CFG), route_monitor=self.route_monitor, env_monitor=self.env_monitor,
        )

    def test_insufficient_data_is_neutral(self):
        adj = self.controller.get_route_adjustment("VWAP_PULLBACK")
        assert adj.multiplier == 1.0
        assert adj.state == "NORMAL"

    def test_strong_route_loosens_thresholds_within_bounds(self):
        for _ in range(15):
            self.route_monitor.record_trade("VWAP_PULLBACK", 100.0)  # all wins -> pf = inf
        adj = self.controller.get_route_adjustment("VWAP_PULLBACK")
        assert adj.state == "FAVORED"
        assert CFG["gate_multiplier_min"] <= adj.multiplier < 1.0

    def test_weak_route_tightens_thresholds_within_bounds(self):
        for i in range(15):
            pnl = 100.0 if i % 4 == 0 else -100.0  # mostly losing -> pf < 1.0
            self.route_monitor.record_trade("PUT_REJECTION", pnl)
        adj = self.controller.get_route_adjustment("PUT_REJECTION")
        assert adj.multiplier > 1.0
        assert adj.multiplier <= CFG["gate_multiplier_max"]

    def test_effective_thresholds_never_exceed_config_bounds(self):
        for _ in range(15):
            self.route_monitor.record_trade("VWAP_PULLBACK", 100.0)
        thresholds = self.controller.get_effective_thresholds("VWAP_PULLBACK")
        # A loosened (FAVORED) route should require a lower score to admit...
        assert thresholds["watch_min_route_score"] < CFG["watch_min_route_score"]
        # ...but never below the configured floor multiplier.
        assert thresholds["watch_min_route_score"] >= (
            CFG["watch_min_route_score"] * CFG["gate_multiplier_min"]
        )
        # Max-style thresholds (confirm_max_adverse_atr) move the other way:
        # loosening a route should *raise* how much adverse move it tolerates.
        assert thresholds["confirm_max_adverse_atr"] > CFG["confirm_max_adverse_atr"]

    def test_spread_quality_keys_are_never_scaled(self):
        """Execution-risk thresholds are deliberately excluded from adaptation."""
        for _ in range(15):
            self.route_monitor.record_trade("VWAP_PULLBACK", 100.0)
        thresholds = self.controller.get_effective_thresholds("VWAP_PULLBACK")
        assert "max_composite_spread_pct_of_mid" not in thresholds
        assert "max_option_mid_inflation" not in thresholds


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE COOLDOWN -> PROBATION RECOVERY
# ═══════════════════════════════════════════════════════════════════════════

class TestRouteRecovery:
    def setup_method(self):
        self.route_monitor = RouteExpectancyMonitor(log_dir="HFT/logs/test_dgc")
        self.env_monitor = EnvironmentExpectancyMonitor(log_dir="HFT/logs/test_dgc")
        self.controller = DynamicGateController(
            config=dict(CFG), route_monitor=self.route_monitor, env_monitor=self.env_monitor,
        )

    def _disable_route(self, route="VWAP_PULLBACK"):
        # Drive profit factor over the 100-trade window below 1.0.
        for i in range(60):
            pnl = 10.0 if i % 5 == 0 else -10.0
            self.route_monitor.record_trade(route, pnl)
        assert self.route_monitor.get_route_status(route) == "DISABLED"

    def test_active_route_passes_through(self):
        allowed, state, size = self.controller.route_trade_allowed("VWAP_PULLBACK")
        assert allowed and state == "ACTIVE" and size == 1.0

    def test_disabled_route_blocked_during_cooldown(self):
        self._disable_route()
        allowed, state, size = self.controller.route_trade_allowed("VWAP_PULLBACK")
        assert not allowed
        assert state == "COOLDOWN"

    def test_disabled_route_recovers_to_probation_after_cooldown(self):
        self._disable_route()
        # Force the cooldown clock to have already elapsed.
        self.controller._route_disabled_since["VWAP_PULLBACK"] = (
            datetime.utcnow() - timedelta(hours=25)
        )
        allowed, state, size = self.controller.route_trade_allowed("VWAP_PULLBACK")
        assert allowed
        assert state == "PROBATION"
        assert size < 1.0

    def test_probation_slots_are_limited_then_exhausted(self):
        self._disable_route()
        self.controller._route_disabled_since["VWAP_PULLBACK"] = (
            datetime.utcnow() - timedelta(hours=25)
        )
        limit = CFG["route_probation_trade_limit"]
        for _ in range(limit):
            allowed, state, _ = self.controller.route_trade_allowed("VWAP_PULLBACK")
            assert allowed and state == "PROBATION"
            self.controller.record_route_probation_trade("VWAP_PULLBACK")

        allowed, state, _ = self.controller.route_trade_allowed("VWAP_PULLBACK")
        assert not allowed
        assert state == "PROBATION_EXHAUSTED"

    def test_route_recovering_to_active_resets_cooldown_state(self):
        self._disable_route()
        self.controller.route_trade_allowed("VWAP_PULLBACK")  # starts the cooldown clock
        assert "VWAP_PULLBACK" in self.controller._route_disabled_since

        # Feed enough winning trades to push pf_50 back up (route recovers
        # once fresh data outweighs the disabled window's bad history).
        for _ in range(30):
            self.route_monitor.record_trade("VWAP_PULLBACK", 100.0)

        allowed, state, size = self.controller.route_trade_allowed("VWAP_PULLBACK")
        assert allowed
        assert state in ("ACTIVE", "SOFT_SIZE_ONLY")
        assert "VWAP_PULLBACK" not in self.controller._route_disabled_since


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT COOLDOWN -> PROBATION RECOVERY
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvironmentRecovery:
    def setup_method(self):
        self.route_monitor = RouteExpectancyMonitor(log_dir="HFT/logs/test_dgc")
        self.env_monitor = EnvironmentExpectancyMonitor(log_dir="HFT/logs/test_dgc")
        self.controller = DynamicGateController(
            config=dict(CFG), route_monitor=self.route_monitor, env_monitor=self.env_monitor,
        )
        self.env = make_env("_ENVTEST")

    def _block_environment(self):
        for _ in range(10):
            self.env_monitor.record_trade(self.env, "AAPL", -10.0)
        assert self.env_monitor.get_environment_status(self.env) == "BLOCKED"

    def test_allowed_environment_passes_through(self):
        allowed, state, size = self.controller.environment_trade_allowed(self.env)
        assert allowed and state == "ALLOWED" and size == 1.0

    def test_blocked_environment_in_cooldown(self):
        self._block_environment()
        allowed, state, size = self.controller.environment_trade_allowed(self.env)
        assert not allowed
        assert state == "COOLDOWN"

    def test_blocked_environment_recovers_to_probation(self):
        self._block_environment()
        self.controller._env_blocked_since[self.env.label] = (
            datetime.utcnow() - timedelta(hours=13)
        )
        allowed, state, size = self.controller.environment_trade_allowed(self.env)
        assert allowed
        assert state == "PROBATION"
        assert size < 1.0
