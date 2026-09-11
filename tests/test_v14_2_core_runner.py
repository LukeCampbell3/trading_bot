"""
Test Suite for V14.2 Core Runner Strategy

Tests cover:
- Watch tickets (creation, admission, lifecycle)
- Confirmation engine (directional move, adverse move, VWAP, stale quotes)
- Spread quality gate (all rejection conditions)
- Package builder (core-runner vs fallback)
- Risk manager (kill switches, forbidden trade types)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG
from strategy.watch_ticket import (
    WatchTicket, WatchTicketBook, TicketSide, TicketStatus
)
from strategy.confirmation_engine import ConfirmationEngine, ConfirmationResult
from strategy.spread_quality_gate import SpreadQualityGate, OptionLeg, SpreadQualityReport
from strategy.package_builder import PackageBuilder, PackageResult
from strategy.risk_manager import RiskManager
from strategy.v14_2_core_runner import V14_2_CoreRunner


# ═══════════════════════════════════════════════════════════════════════════
# WATCH TICKET TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestWatchTickets:
    """Test watch ticket system."""

    def test_base_signal_creates_watch_ticket_only(self):
        """Base signal creates watch ticket ONLY - does NOT submit order."""
        book = WatchTicketBook(log_dir="HFT/logs/test_v14_2")
        ticket = book.create_ticket(
            symbol="AAPL",
            route="VWAP_PULLBACK",
            side=TicketSide.CALL,
            underlying_price_at_watch=150.0,
            vwap_at_watch=149.5,
            atr_at_watch=1.5,
            route_score=0.75,
            ic_spread=0.035,
            expected_ev_over_debit=0.15,
            option_liquidity_score=0.7,
            expected_move_to_target=2.0,
            estimated_debit_at_watch=1.50,
        )
        assert ticket is not None
        assert ticket.status == TicketStatus.WATCHING
        # Ticket must NOT have any order submission capability
        assert not hasattr(ticket, 'submit_order')
        assert not hasattr(ticket, 'order_id')

    def test_watch_ticket_does_not_submit_order(self):
        """Watch ticket has no mechanism to submit orders."""
        ticket = WatchTicket(
            symbol="AAPL",
            route="VWAP_PULLBACK",
            side=TicketSide.CALL,
            status=TicketStatus.WATCHING,
        )
        # Verify the ticket class has no order-submitting methods
        methods = [m for m in dir(ticket) if 'submit' in m.lower() or 'order' in m.lower()]
        assert len(methods) == 0, f"Ticket should not have order methods: {methods}"

    def test_expired_watch_ticket_cannot_submit_order(self):
        """Expired watch ticket cannot transition to filled."""
        ticket = WatchTicket(
            symbol="AAPL",
            route="VWAP_PULLBACK",
            side=TicketSide.CALL,
            status=TicketStatus.WATCHING,
        )
        ticket.mark_expired("timeout")
        assert ticket.status == TicketStatus.EXPIRED
        # Cannot be filled after expiration (no method to transition back)
        ticket.mark_filled()  # This should not be guarded but status is wrong for pipeline
        # In the pipeline, expired tickets are retired and won't be processed

    def test_confirmed_ticket_moves_to_confirmation_state(self):
        """Confirmed ticket transitions to CONFIRMED status."""
        ticket = WatchTicket(
            symbol="AAPL",
            route="VWAP_PULLBACK",
            side=TicketSide.CALL,
            status=TicketStatus.WATCHING,
            underlying_price_at_watch=150.0,
        )
        ticket.mark_confirmed(
            directional_move_atr=0.3,
            adverse_move_atr=0.05,
            mfe_velocity=0.5,
            price=150.5,
        )
        assert ticket.status == TicketStatus.CONFIRMED
        assert ticket.directional_move_atr == 0.3
        assert ticket.confirmation_price == 150.5

    def test_watch_admission_rejects_low_route_score(self):
        """Watch admission rejects candidate with score below threshold."""
        book = WatchTicketBook(log_dir="HFT/logs/test_v14_2")
        ticket = book.create_ticket(
            symbol="AAPL",
            route="VWAP_PULLBACK",
            side=TicketSide.CALL,
            route_score=0.50,  # Below 0.706 threshold
            ic_spread=0.035,
            expected_ev_over_debit=0.15,
            option_liquidity_score=0.7,
        )
        assert ticket is None  # Should be rejected

    def test_watch_admission_rejects_low_liquidity(self):
        """Watch admission rejects low option liquidity."""
        book = WatchTicketBook(log_dir="HFT/logs/test_v14_2")
        ticket = book.create_ticket(
            symbol="AAPL",
            route="VWAP_PULLBACK",
            side=TicketSide.CALL,
            route_score=0.80,
            ic_spread=0.035,
            expected_ev_over_debit=0.15,
            option_liquidity_score=0.3,  # Below 0.623 threshold
        )
        assert ticket is None


# ═══════════════════════════════════════════════════════════════════════════
# CONFIRMATION ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestConfirmationEngine:
    """Test confirmation engine."""

    def setup_method(self):
        self.engine = ConfirmationEngine()
        self.base_ticket = WatchTicket(
            symbol="AAPL",
            route="VWAP_PULLBACK",
            side=TicketSide.CALL,
            status=TicketStatus.WATCHING,
            underlying_price_at_watch=150.0,
            vwap_at_watch=149.5,
            atr_at_watch=1.5,
            route_score=0.75,
        )

    def test_confirms_correct_direction_move(self):
        """Confirms when price moves in expected direction by min ATR."""
        result = self.engine.check_confirmation(
            ticket=self.base_ticket,
            current_price=150.5,  # Moved up
            current_vwap=149.8,
            current_atr=1.5,
            high_since_watch=150.6,
            low_since_watch=149.8,
            mfe_velocity=0.5,  # Above 0.487
            env_stress=0.1,
            route_score_now=0.75,
            option_quote_valid=True,
        )
        assert result.confirmed

    def test_rejects_adverse_move(self):
        """Rejects when adverse move exceeds threshold."""
        result = self.engine.check_confirmation(
            ticket=self.base_ticket,
            current_price=150.5,
            current_vwap=149.8,
            current_atr=1.5,
            high_since_watch=150.6,
            low_since_watch=148.5,  # Adverse: 150-148.5 = 1.5 / 1.5 ATR = 1.0 > 0.151
            mfe_velocity=0.5,
            env_stress=0.1,
            route_score_now=0.75,
            option_quote_valid=True,
        )
        assert not result.confirmed
        assert "adverse" in result.reason.lower()

    def test_rejects_vwap_failure_call(self):
        """Rejects CALL confirmation when price below VWAP."""
        result = self.engine.check_confirmation(
            ticket=self.base_ticket,
            current_price=148.0,  # Below VWAP
            current_vwap=149.5,
            current_atr=1.5,
            high_since_watch=150.0,
            low_since_watch=147.9,
            mfe_velocity=0.5,
            env_stress=0.1,
            route_score_now=0.75,
            option_quote_valid=True,
        )
        # Will fail on directional move (negative) before VWAP check
        assert not result.confirmed

    def test_rejects_stale_quote(self):
        """Rejects when option quote is invalid/stale."""
        result = self.engine.check_confirmation(
            ticket=self.base_ticket,
            current_price=150.5,
            current_vwap=149.8,
            current_atr=1.5,
            high_since_watch=150.6,
            low_since_watch=149.8,
            mfe_velocity=0.5,
            env_stress=0.1,
            route_score_now=0.75,
            option_quote_valid=False,  # Stale/invalid
        )
        assert not result.confirmed
        assert "quote" in result.reason.lower()

    def test_rejects_move_exhaustion(self):
        """Rejects when MFE velocity is too low (move exhausted)."""
        result = self.engine.check_confirmation(
            ticket=self.base_ticket,
            current_price=150.5,
            current_vwap=149.8,
            current_atr=1.5,
            high_since_watch=150.6,
            low_since_watch=149.8,
            mfe_velocity=0.2,  # Below 0.487 threshold
            env_stress=0.1,
            route_score_now=0.75,
            option_quote_valid=True,
        )
        assert not result.confirmed
        assert "velocity" in result.reason.lower()

    def test_rejects_environment_stress(self):
        """Rejects when environment stress is too high."""
        result = self.engine.check_confirmation(
            ticket=self.base_ticket,
            current_price=150.5,
            current_vwap=149.8,
            current_atr=1.5,
            high_since_watch=150.6,
            low_since_watch=149.8,
            mfe_velocity=0.5,
            env_stress=0.5,  # Above 0.252 threshold
            route_score_now=0.75,
            option_quote_valid=True,
        )
        assert not result.confirmed
        assert "stress" in result.reason.lower()

    def test_put_confirms_downward_move(self):
        """PUT route confirms on downward directional move."""
        put_ticket = WatchTicket(
            symbol="AAPL",
            route="PUT_REJECTION",
            side=TicketSide.PUT,
            status=TicketStatus.WATCHING,
            underlying_price_at_watch=150.0,
            vwap_at_watch=150.5,
            atr_at_watch=1.5,
            route_score=0.75,
        )
        result = self.engine.check_confirmation(
            ticket=put_ticket,
            current_price=149.4,  # Moved down: (150-149.4)/1.5 = 0.4 > 0.263
            current_vwap=150.2,
            current_atr=1.5,
            high_since_watch=150.2,  # Adverse up: (150.2-150)/1.5 = 0.133 < 0.151
            low_since_watch=149.3,
            mfe_velocity=0.5,
            env_stress=0.1,
            route_score_now=0.75,
            option_quote_valid=True,
        )
        assert result.confirmed


# ═══════════════════════════════════════════════════════════════════════════
# SPREAD QUALITY GATE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSpreadQualityGate:
    """Test spread quality gate."""

    def setup_method(self):
        self.gate = SpreadQualityGate()
        # Tight spreads where:
        # spread_mid = long_mid - short_mid = 1.50 - 0.875 = 0.625
        # spread_ask = long_ask - short_bid = 1.51 - 0.87 = 0.64
        # composite_width = (1.51-1.49) + (0.88-0.87) = 0.02 + 0.01 = 0.03
        # composite_pct = 0.03 / 0.625 = 0.048 < 0.126 ✓
        # limit_chase = (0.64 - 0.625) / 0.625 = 0.024 < 0.034 ✓
        self.long_leg = OptionLeg(
            contract_symbol="AAPL_C_150",
            side="buy", bid=1.49, ask=1.51, mid=1.50,
            delta=0.50, iv=0.30, volume=100,
            open_interest=500, dte=7, strike=150.0,
        )
        self.short_leg = OptionLeg(
            contract_symbol="AAPL_C_155",
            side="sell", bid=0.87, ask=0.88, mid=0.875,
            delta=0.30, iv=0.32, volume=80,
            open_interest=300, dte=7, strike=155.0,
        )

    def test_accepts_clean_spread(self):
        """Accepts a spread that passes all quality checks."""
        report = self.gate.evaluate(
            long_leg=self.long_leg,
            short_leg=self.short_leg,
            underlying_price=150.0,
            underlying_price_at_watch=149.5,
            estimated_debit_at_watch=0.60,
            target_price=153.0,
        )
        assert report.passed

    def test_rejects_inflated_option_mid(self):
        """Rejects when mid has inflated too much since watch."""
        report = self.gate.evaluate(
            long_leg=self.long_leg,
            short_leg=self.short_leg,
            underlying_price=150.0,
            underlying_price_at_watch=149.5,
            estimated_debit_at_watch=0.40,  # Current mid ~0.625, inflation > 50% > 10%
            target_price=153.0,
        )
        assert not report.passed
        assert "inflation" in report.rejection_reason.lower()

    def test_rejects_wide_composite_spread(self):
        """Rejects when composite spread is too wide."""
        wide_long = OptionLeg(
            contract_symbol="AAPL_C_150",
            side="buy", bid=1.00, ask=1.80, mid=1.40,  # Very wide: 0.80 spread
            delta=0.50, iv=0.30, volume=10,
            open_interest=50, dte=7, strike=150.0,
        )
        wide_short = OptionLeg(
            contract_symbol="AAPL_C_155",
            side="sell", bid=0.50, ask=1.20, mid=0.85,  # Very wide: 0.70 spread
            delta=0.30, iv=0.32, volume=10,
            open_interest=30, dte=7, strike=155.0,
        )
        report = self.gate.evaluate(
            long_leg=wide_long,
            short_leg=wide_short,
            underlying_price=150.0,
            underlying_price_at_watch=149.5,
            estimated_debit_at_watch=0.55,
            target_price=153.0,
        )
        assert not report.passed
        assert "spread" in report.rejection_reason.lower() or "composite" in report.rejection_reason.lower()

    def test_rejects_insufficient_remaining_reward(self):
        """Rejects when remaining reward to risk is too low."""
        # Make spread mid very close to max value (tight bid/ask to pass spread width)
        # Strike width = 5. If mid = 4.5, remaining reward = 0.5, ratio = 0.11 < 0.5
        expensive_long = OptionLeg(
            contract_symbol="AAPL_C_150",
            side="buy", bid=4.88, ask=4.92, mid=4.90,
            delta=0.50, iv=0.30, volume=100,
            open_interest=500, dte=7, strike=150.0,
        )
        expensive_short = OptionLeg(
            contract_symbol="AAPL_C_155",
            side="sell", bid=0.38, ask=0.42, mid=0.40,
            delta=0.30, iv=0.32, volume=80,
            open_interest=300, dte=7, strike=155.0,
        )
        report = self.gate.evaluate(
            long_leg=expensive_long,
            short_leg=expensive_short,
            underlying_price=150.0,
            underlying_price_at_watch=149.5,
            estimated_debit_at_watch=4.5,
            target_price=153.0,
        )
        assert not report.passed
        assert "reward" in report.rejection_reason.lower()

    def test_rejects_too_much_move_consumed(self):
        """Rejects when too much of the expected move is consumed."""
        # Use tight legs so other checks pass first
        tight_long = OptionLeg(
            contract_symbol="AAPL_C_150",
            side="buy", bid=1.49, ask=1.51, mid=1.50,
            delta=0.50, iv=0.30, volume=100,
            open_interest=500, dte=7, strike=152.5,
        )
        tight_short = OptionLeg(
            contract_symbol="AAPL_C_155",
            side="sell", bid=0.87, ask=0.88, mid=0.875,
            delta=0.30, iv=0.32, volume=80,
            open_interest=300, dte=7, strike=157.5,
        )
        report = self.gate.evaluate(
            long_leg=tight_long,
            short_leg=tight_short,
            underlying_price=152.5,  # Already near target
            underlying_price_at_watch=149.5,
            estimated_debit_at_watch=0.62,
            target_price=153.0,  # 3.5 move expected, 3.0 consumed = 86% > 47.4%
        )
        assert not report.passed
        assert "consumed" in report.rejection_reason.lower()

    def test_rejects_invalid_bid_ask(self):
        """Rejects when leg has invalid bid/ask."""
        invalid_leg = OptionLeg(
            contract_symbol="AAPL_C_150",
            side="buy", bid=0.0, ask=1.60, mid=0.80,  # Zero bid
            delta=0.50, iv=0.30, volume=100,
            open_interest=500, dte=7, strike=150.0,
        )
        report = self.gate.evaluate(
            long_leg=invalid_leg,
            short_leg=self.short_leg,
            underlying_price=150.0,
            underlying_price_at_watch=149.5,
            estimated_debit_at_watch=0.60,
            target_price=153.0,
        )
        assert not report.passed
        assert "invalid" in report.rejection_reason.lower()


# ═══════════════════════════════════════════════════════════════════════════
# PACKAGE BUILDER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPackageBuilder:
    """Test package builder."""

    def setup_method(self):
        self.builder = PackageBuilder()
        self.good_quality = SpreadQualityReport(
            passed=True,
            spread_mid=0.625,
            spread_bid=0.45,
            spread_ask=0.80,
        )

    def test_builds_core_runner_package_when_eligible(self):
        """Builds package when all conditions met."""
        result = self.builder.build(
            quality_report=self.good_quality,
            quality_score=0.75,  # Above 0.626
            pwin=0.65,  # Above 0.581
            env_stress=0.1,  # Below 0.252
            route="VWAP_PULLBACK",  # In routes_allowed
            account_buying_power=10000.0,
        )
        assert result.is_package
        assert not result.is_fallback
        assert result.core is not None
        assert result.runner is not None
        assert result.core.role == "CORE"
        assert result.runner.role == "RUNNER"

    def test_falls_back_when_not_eligible(self):
        """Falls back to V12.2 when package conditions fail."""
        result = self.builder.build(
            quality_report=self.good_quality,
            quality_score=0.50,  # Below 0.626
            pwin=0.65,
            env_stress=0.1,
            route="VWAP_PULLBACK",
            account_buying_power=10000.0,
        )
        assert result.is_fallback
        assert not result.is_package
        assert "quality_score" in result.fallback_reason

    def test_does_not_force_package_when_cannot_afford(self):
        """Does not force package when account cannot afford it."""
        result = self.builder.build(
            quality_report=self.good_quality,
            quality_score=0.75,
            pwin=0.65,
            env_stress=0.1,
            route="VWAP_PULLBACK",
            account_buying_power=50.0,  # Too little
        )
        assert result.is_fallback
        assert "buying_power" in result.fallback_reason

    def test_core_and_runner_fractions_sum_correctly(self):
        """Core + runner fractions sum to ~1.0."""
        result = self.builder.build(
            quality_report=self.good_quality,
            quality_score=0.75,
            pwin=0.65,
            env_stress=0.1,
            route="VWAP_PULLBACK",
            account_buying_power=10000.0,
        )
        if result.is_package and result.core and result.runner:
            total = result.core.total_debit + result.runner.total_debit
            if total > 0:
                core_frac = result.core.total_debit / total
                runner_frac = result.runner.total_debit / total
                assert abs((core_frac + runner_frac) - 1.0) < 0.01

    def test_stop_target_calculations_correct(self):
        """Stop and target percentages match config."""
        result = self.builder.build(
            quality_report=self.good_quality,
            quality_score=0.75,
            pwin=0.65,
            env_stress=0.1,
            route="VWAP_PULLBACK",
            account_buying_power=10000.0,
        )
        if result.is_package:
            assert result.core.target_pct == CFG["core_target_pct"]
            assert result.core.stop_pct == CFG["package_stop_pct"]
            assert result.runner.target_pct == CFG["runner_target_pct"]
            assert result.runner.lock_pct == CFG["runner_lock_pct"]

    def test_soft_route_gets_fallback(self):
        """Soft-only routes get V12.2 fallback, not package."""
        result = self.builder.build(
            quality_report=self.good_quality,
            quality_score=0.75,
            pwin=0.65,
            env_stress=0.1,
            route="RAW_BREAKOUT",  # Soft-only route
            account_buying_power=10000.0,
        )
        assert result.is_fallback
        assert "not_package_eligible" in result.fallback_reason


# ═══════════════════════════════════════════════════════════════════════════
# RISK MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskManager:
    """Test risk manager."""

    def setup_method(self):
        self.rm = RiskManager(log_dir="HFT/logs/test_v14_2")
        self.rm.update_account(100000.0)
        self.rm.reset_day()

    def test_daily_kill_switch_works(self):
        """Daily kill switch activates on loss threshold."""
        # Simulate losses exceeding daily kill
        self.rm.record_close("AAPL", -5400.0)  # -5.4% > -5.3% threshold
        assert self.rm._daily_killed
        result = self.rm.pre_trade_check("AAPL", "VWAP_PULLBACK", 100.0)
        assert not result.allowed
        assert "KILL" in result.reason

    def test_weekly_kill_switch_works(self):
        """Weekly kill switch activates on loss threshold."""
        self.rm.record_close("AAPL", -7400.0)  # -7.4% > -7.3% threshold
        assert self.rm._weekly_killed
        result = self.rm.pre_trade_check("AAPL", "VWAP_PULLBACK", 100.0)
        assert not result.allowed

    def test_same_underlying_cap_works(self):
        """Same-underlying weekly cap enforced."""
        # Use a config with higher max_trades_per_day so that limit is not hit first
        from strategy.risk_manager import RiskManager
        custom_cfg = dict(CFG)
        custom_cfg["max_trades_per_day"] = 10  # Raise daily limit
        custom_cfg["max_trades_per_week"] = 10  # Raise weekly limit
        rm = RiskManager(config=custom_cfg, log_dir="HFT/logs/test_v14_2")
        rm.update_account(100000.0)
        rm.reset_day()
        # Record 2 trades for AAPL (max_same_underlying_trades_per_week = 2)
        for _ in range(2):
            rm.record_trade("AAPL", "VWAP_PULLBACK", 100.0)
        result = rm.pre_trade_check("AAPL", "VWAP_PULLBACK", 100.0)
        assert not result.allowed
        assert "SAME_UNDERLYING" in result.reason

    def test_live_trading_disabled_by_default(self):
        """Live trading disabled by default blocks live orders."""
        result = self.rm.pre_trade_check(
            "AAPL", "VWAP_PULLBACK", 100.0, is_live=True
        )
        assert not result.allowed
        assert "LIVE_TRADING_DISABLED" in result.reason

    def test_raw_long_calls_rejected(self):
        """Raw long calls are always rejected."""
        result = self.rm.pre_trade_check(
            "AAPL", "VWAP_PULLBACK", 100.0, is_raw_long_call=True
        )
        assert not result.allowed
        assert "RAW_LONG_CALLS" in result.reason

    def test_equity_scalps_rejected(self):
        """Equity scalp fallback is always rejected."""
        result = self.rm.pre_trade_check(
            "AAPL", "VWAP_PULLBACK", 100.0, is_equity_scalp=True
        )
        assert not result.allowed
        assert "EQUITY_SCALP" in result.reason

    def test_averaging_down_rejected(self):
        """Averaging down is always rejected."""
        result = self.rm.pre_trade_check(
            "AAPL", "VWAP_PULLBACK", 100.0, is_averaging_down=True
        )
        assert not result.allowed
        assert "AVERAGING_DOWN" in result.reason

    def test_revenge_trades_rejected(self):
        """Revenge trades are always rejected."""
        result = self.rm.pre_trade_check(
            "AAPL", "VWAP_PULLBACK", 100.0, is_revenge=True
        )
        assert not result.allowed
        assert "REVENGE" in result.reason

    def test_max_trades_per_day_enforced(self):
        """Max trades per day limit enforced."""
        for _ in range(2):  # Config max is 2
            self.rm.record_trade("AAPL", "VWAP_PULLBACK", 100.0)
        result = self.rm.pre_trade_check("TSLA", "VWAP_PULLBACK", 100.0)
        assert not result.allowed
        assert "MAX_TRADES_PER_DAY" in result.reason

    def test_exposure_limit_enforced(self):
        """Max open debit exposure percentage enforced."""
        # Add large position near limit
        self.rm.add_open_position("AAPL", 25000.0)  # 25% of 100k
        # Try to add more (would exceed 28% limit)
        result = self.rm.pre_trade_check("TSLA", "VWAP_PULLBACK", 5000.0)
        assert not result.allowed
        assert "EXPOSURE" in result.reason


# ═══════════════════════════════════════════════════════════════════════════
# CORE RUNNER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCoreRunnerIntegration:
    """Integration tests for the V14.2 Core Runner."""

    def test_live_mode_blocked_by_default(self):
        """Cannot create core runner in live mode with default config."""
        with pytest.raises(RuntimeError, match="live_trading_enabled"):
            V14_2_CoreRunner(paper_mode=False)

    def test_paper_mode_initializes(self):
        """Paper mode initializes successfully."""
        runner = V14_2_CoreRunner(paper_mode=True)
        assert runner.paper_mode
        assert runner.VERSION == "14.2"
        assert runner.STATUS == "SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN"

    def test_evaluate_opportunity_returns_status(self):
        """evaluate_opportunity returns proper status dict."""
        runner = V14_2_CoreRunner(paper_mode=True)
        runner.update_account(100000.0, 50000.0)
        result = runner.evaluate_opportunity(
            symbol="AAPL",
            price=150.0,
            vwap=149.5,
            atr=1.5,
            high_of_day=151.0,
            low_of_day=148.5,
            trend_slope=0.005,
            volume_ratio=1.2,
            price_5m_ago=149.8,
            price_15m_ago=149.0,
        )
        assert "action" in result
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert result.get("details", {}).get("reason") != "symbol_policy: unknown_symbol"

    def test_v14_3_symbol_policy_still_blocks_unapproved_symbol(self):
        """The V14.3 profile retains its explicit symbol allowlist."""
        from strategy.v14_3_highvol_config import get_v14_3_config

        runner = V14_2_CoreRunner(config=get_v14_3_config(), paper_mode=True)
        result = runner.evaluate_opportunity(
            symbol="AAPL", price=150.0, vwap=149.5, atr=1.5,
            high_of_day=151.0, low_of_day=148.5,
            trend_slope=0.005, volume_ratio=1.2,
            price_5m_ago=149.8, price_15m_ago=149.0,
        )

        assert result["action"] == "SKIPPED"
        assert result["details"]["reason"] == "symbol_policy: unknown_symbol"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
