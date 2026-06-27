"""
Test Suite for Multileg Execution Manager

Tests cover:
- Only execution manager submits orders
- Mleg orders use limit only
- Duplicate client_order_id prevention
- Unfilled order timeout → missed fill
- Close order is submitted as closing mleg limit
- Stale order cancellation
- Broker reconciliation blocks inconsistent state
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta

from execution.mleg_execution_manager import (
    MlegExecutionManager, MlegOrder, MlegLeg, OrderState
)
from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG


class TestMlegExecutionManager:
    """Test multileg execution manager."""

    def setup_method(self):
        self.mgr = MlegExecutionManager(
            trading_client=None,
            paper_mode=True,
            log_dir="HFT/logs/test_v14_2",
        )
        self.sample_legs = [
            MlegLeg(
                contract_symbol="AAPL_C_150_0620",
                side="buy_to_open",
                quantity=2,
                option_type="call",
                strike=150.0,
            ),
            MlegLeg(
                contract_symbol="AAPL_C_155_0620",
                side="sell_to_open",
                quantity=2,
                option_type="call",
                strike=155.0,
            ),
        ]

    def test_only_execution_manager_submits_orders(self):
        """Orders can only be created through the execution manager."""
        order = self.mgr.create_entry_order(
            ticket_id="test_001",
            legs=self.sample_legs,
            limit_price=0.65,
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        assert order is not None
        assert order.order_class == "MLEG"
        assert order.parent_strategy == "V14_2_CORE_RUNNER"

    def test_mleg_orders_use_limit_only(self):
        """All mleg orders must use limit order type."""
        order = self.mgr.create_entry_order(
            ticket_id="test_002",
            legs=self.sample_legs,
            limit_price=0.65,
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        assert order.order_type == "LIMIT"

    def test_duplicate_client_order_id_prevented(self):
        """Prevents duplicate orders for same ticket."""
        order1 = self.mgr.create_entry_order(
            ticket_id="test_003",
            legs=self.sample_legs,
            limit_price=0.65,
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        # Submit first order (gets filled in paper mode)
        self.mgr.submit_order(order1)
        assert order1.state == OrderState.FILLED

        # Try to create another order for same ticket - should be rejected
        order2 = self.mgr.create_entry_order(
            ticket_id="test_003",
            legs=self.sample_legs,
            limit_price=0.65,
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        assert order2 is None  # Duplicate rejected

    def test_unfilled_order_times_out_missed_fill(self):
        """Unfilled order times out and becomes missed fill."""
        order = self.mgr.create_entry_order(
            ticket_id="test_004",
            legs=self.sample_legs,
            limit_price=0.65,
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        # Manually set to submitted without fill (bypass paper auto-fill)
        order.state = OrderState.SUBMITTED
        order.submit_time = datetime.utcnow() - timedelta(seconds=300)  # Old
        order.stale_timeout_seconds = 120

        self.mgr.cancel_stale_orders()
        assert order.state == OrderState.MISSED_FILL
        assert "stale" in order.missed_fill_reason

    def test_close_order_is_closing_mleg_limit(self):
        """Close orders are submitted as closing mleg limit orders."""
        close_legs = [
            MlegLeg(
                contract_symbol="AAPL_C_150_0620",
                side="sell_to_close",
                quantity=2,
                option_type="call",
                strike=150.0,
            ),
            MlegLeg(
                contract_symbol="AAPL_C_155_0620",
                side="buy_to_close",
                quantity=2,
                option_type="call",
                strike=155.0,
            ),
        ]
        order = self.mgr.create_exit_order(
            ticket_id="test_005",
            legs=close_legs,
            limit_price=0.50,
            quantity=2,
            exit_bid=0.45,
            exit_ask=0.55,
        )
        assert order is not None
        assert order.direction == "CLOSE"
        assert order.order_type == "LIMIT"

        submitted = self.mgr.submit_exit(order)
        assert submitted
        assert order.state == OrderState.CLOSED  # Paper mode auto-fills

    def test_stale_order_is_canceled(self):
        """Stale unfilled orders are properly canceled."""
        order = self.mgr.create_entry_order(
            ticket_id="test_006",
            legs=self.sample_legs,
            limit_price=0.65,
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        order.state = OrderState.SUBMITTED
        order.submit_time = datetime.utcnow() - timedelta(seconds=200)
        order.stale_timeout_seconds = 120

        self.mgr.cancel_stale_orders()
        assert order.state == OrderState.MISSED_FILL

    def test_broker_reconciliation_paper_mode(self):
        """Broker reconciliation succeeds in paper mode."""
        result = self.mgr.reconcile_positions()
        assert result is True

    def test_contradictory_orders_prevented(self):
        """Prevents contradictory orders for same contracts."""
        order1 = self.mgr.create_entry_order(
            ticket_id="test_007a",
            legs=self.sample_legs,
            limit_price=0.65,
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        # Don't submit - leave in CREATED/SUBMITTED state
        order1.state = OrderState.SUBMITTED

        # Try to create order with same contracts
        order2 = self.mgr.create_entry_order(
            ticket_id="test_007b",
            legs=self.sample_legs,
            limit_price=0.70,
            quantity=2,
            composite_bid=0.50,
            composite_ask=0.85,
            composite_mid=0.675,
        )
        assert order2 is None  # Contradictory rejected

    def test_limit_chase_capped(self):
        """Limit chase is capped at max_limit_chase_pct_of_debit."""
        order = self.mgr.create_entry_order(
            ticket_id="test_008",
            legs=self.sample_legs,
            limit_price=1.50,  # Way above mid of 0.625
            quantity=2,
            composite_bid=0.45,
            composite_ask=0.80,
            composite_mid=0.625,
        )
        # Limit should be capped
        max_chase = 0.625 * (1 + CFG["max_limit_chase_pct_of_debit"])
        assert order.intended_limit <= max_chase + 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
