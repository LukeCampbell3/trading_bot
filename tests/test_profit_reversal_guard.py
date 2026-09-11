"""Regression tests for profit ratcheting and reversal-safe execution."""

import unittest
from datetime import datetime

from execution_controller import (
    BrokerSymbolState,
    Decision,
    ExecutionPolicy,
    FreshMarketSignalEngine,
    LossGovernor,
    SafetyMode,
    SymbolCampaign,
    TradeSignal,
    ExecutionReconciler,
    _decision_allowed_by_safety,
)


class _NoopBroker:
    def cancel_order(self, order_id):
        return None

    def submit_exit_order(self, **kwargs):
        return True

    def submit_entry_order(self, **kwargs):
        return True


class _SafetyStub:
    def __init__(self, mode, allow_entries, allow_exits=True):
        self.mode = mode
        self._allow_entries = allow_entries
        self._allow_exits = allow_exits

    def allows_new_entries(self):
        return self._allow_entries

    def allows_exits(self):
        return self._allow_exits


class _SignalEngine:
    base_exit_threshold = 0.00010
    signal_reversal_delta = 0.000015

    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def generate_signals(self):
        value = self.values[self.index]
        self.index += 1
        return [
            TradeSignal(
                symbol="AAPL",
                bucket_id="B3",
                direction="long" if value > 0 else "flat",
                score=0.8,
                expected_return_bps=value * 10000.0,
                setup_fingerprint="fp",
                generated_at=datetime.utcnow(),
                ttl_seconds=90,
                reason=f"sig={value:.6f}",
            )
        ]

    def get_market_state(self, symbol):
        raise AssertionError("not used")

    def estimate_costs(self, symbol):
        raise AssertionError("not used")


class ProfitRatchetTests(unittest.TestCase):
    def setUp(self):
        self.policy = ExecutionPolicy()
        self.governor = LossGovernor(self.policy)

    def _campaign(self):
        return SymbolCampaign(
            symbol="AAPL",
            state="ACTIVE",
            target_qty=10.0,
            entry_price=100.0,
            setup_fingerprint="fp",
        )

    def test_peak_profit_creates_ratcheting_exit_floor(self):
        campaign = self._campaign()

        action = self.governor.evaluate_profit_protection(campaign, 101.30)
        self.assertEqual(action, "protect_profit")
        self.governor.apply_profit_action(campaign, action)
        self.assertEqual(campaign.state, "PROTECT_PROFIT")
        first_floor = campaign._profit_floor_pct
        self.assertGreaterEqual(first_floor, 0.0075)

        action = self.governor.evaluate_profit_protection(campaign, 102.50)
        self.assertEqual(action, "trim_trail")
        self.governor.apply_profit_action(campaign, action)
        self.assertEqual(campaign.state, "REDUCING")
        self.assertEqual(campaign.target_qty, 5.0)
        self.assertTrue(campaign._profit_trimmed)
        second_floor = campaign._profit_floor_pct
        self.assertGreater(second_floor, first_floor)

        # A pullback below the ratcheted floor must flatten the remainder.
        action = self.governor.evaluate_profit_protection(campaign, 101.50)
        self.assertEqual(action, "trail_exit")
        self.governor.apply_profit_action(campaign, action)
        self.assertEqual(campaign.target_qty, 0)
        self.assertEqual(campaign.state, "EXITING")
        self.assertEqual(campaign.last_exit_reason, "profit_trail")

    def test_profit_trim_only_happens_once(self):
        campaign = self._campaign()

        action = self.governor.evaluate_profit_protection(campaign, 102.20)
        self.assertEqual(action, "trim_trail")
        self.governor.apply_profit_action(campaign, action)
        self.assertEqual(campaign.target_qty, 5.0)

        # Same or higher profit cannot repeatedly halve target size.
        action = self.governor.evaluate_profit_protection(campaign, 102.30)
        self.assertNotEqual(action, "trim_trail")
        self.assertEqual(campaign.target_qty, 5.0)

    def test_completed_reduction_returns_to_profit_protection(self):
        campaign = self._campaign()
        campaign.state = "REDUCING"
        campaign.target_qty = 5.0
        campaign._profit_trimmed = True

        broker_state = BrokerSymbolState(
            symbol="AAPL",
            position_qty=5,
            avg_entry_price=100.0,
            pending_buy_qty=0,
            pending_sell_qty=0,
        )
        reconciler = ExecutionReconciler(
            _NoopBroker(),
            self.policy,
            eastern_tz=None,
        )

        result = reconciler.reconcile_symbol(
            "AAPL", campaign, broker_state
        )
        self.assertEqual(result, "reduction_complete")
        self.assertEqual(campaign.state, "PROTECT_PROFIT")

    def test_pending_sell_does_not_fake_reduction_completion(self):
        campaign = self._campaign()
        campaign.state = "REDUCING"
        campaign.target_qty = 5.0
        campaign._profit_trimmed = True

        broker_state = BrokerSymbolState(
            symbol="AAPL",
            position_qty=10,
            avg_entry_price=100.0,
            pending_buy_qty=0,
            pending_sell_qty=5,
            open_sell_order_ids=["sell-1"],
        )
        reconciler = ExecutionReconciler(
            _NoopBroker(),
            self.policy,
            eastern_tz=None,
        )

        result = reconciler.reconcile_symbol(
            "AAPL", campaign, broker_state
        )
        self.assertEqual(result, "in_sync")
        self.assertEqual(campaign.state, "REDUCING")


class ReversalGuardTests(unittest.TestCase):
    def test_weakening_positive_signal_flattens_before_crossing_zero(self):
        # 5.0 bps -> 0.8 bps.  Existing v14.2 exit knobs imply a 1.0 bps
        # weak threshold and 0.15 bps minimum deterioration.
        engine = _SignalEngine([0.00050, 0.00008])
        guard = FreshMarketSignalEngine(engine, refresh_interval_seconds=0)

        first = guard.generate_signals()[0]
        second = guard.generate_signals()[0]

        self.assertEqual(first.direction, "long")
        self.assertEqual(second.direction, "flat")
        self.assertIn("reversal_guard", second.reason)

    def test_strong_positive_signal_is_not_forced_flat(self):
        engine = _SignalEngine([0.00050, 0.00020])
        guard = FreshMarketSignalEngine(engine, refresh_interval_seconds=0)

        guard.generate_signals()
        second = guard.generate_signals()[0]

        self.assertEqual(second.direction, "long")


class SafetyExitTests(unittest.TestCase):
    def test_no_new_entries_mode_still_allows_flatten(self):
        manager = _SafetyStub(
            SafetyMode.NO_NEW_ENTRIES,
            allow_entries=False,
            allow_exits=True,
        )
        self.assertTrue(
            _decision_allowed_by_safety(
                Decision(action="TARGET_ZERO"), manager
            )
        )
        self.assertFalse(
            _decision_allowed_by_safety(
                Decision(action="SET_TARGET", target_qty=10), manager
            )
        )

    def test_reconciliation_only_blocks_strategy_orders(self):
        manager = _SafetyStub(
            SafetyMode.ORDER_RECONCILIATION_ONLY,
            allow_entries=False,
            allow_exits=True,
        )
        self.assertFalse(
            _decision_allowed_by_safety(
                Decision(action="TARGET_ZERO"), manager
            )
        )
        self.assertFalse(
            _decision_allowed_by_safety(
                Decision(action="SET_TARGET", target_qty=10), manager
            )
        )


if __name__ == "__main__":
    unittest.main()
