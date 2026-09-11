"""Regression tests for broker-backed V14.3 options integration."""

from types import SimpleNamespace

import pytest

from execution.mleg_execution_manager import MlegLeg, OrderState
from execution.v14_3_mleg_execution_manager import V14_3_MlegExecutionManager
from strategy.route_conditioning import RouteCandidate
from strategy.v14_3_highvol_config import get_v14_3_config
from strategy.v14_3_options_runner import V14_3_OptionsRunner


class _RejectingBroker:
    def submit_order(self, order_data=None):
        raise RuntimeError("upstream unavailable")

    def get_all_positions(self):
        return []


class _CapturingBroker:
    def __init__(self):
        self.request = None

    def submit_order(self, order_data=None):
        self.request = order_data
        return SimpleNamespace(id="paper-order-1")

    def get_order_by_id(self, order_id):
        return SimpleNamespace(status="new", filled_avg_price=None)

    def get_all_positions(self):
        return []


class _DummyOptionData:
    pass


def _entry(manager):
    return manager.create_entry_order(
        ticket_id="ticket-1",
        legs=[
            MlegLeg("COIN261016C00200000", "buy_to_open", 1, "call", 200.0, "2026-10-16"),
            MlegLeg("COIN261016C00205000", "sell_to_open", 1, "call", 205.0, "2026-10-16"),
        ],
        limit_price=1.25,
        quantity=1,
        composite_bid=1.10,
        composite_ask=1.30,
        composite_mid=1.20,
    )


def test_broker_failure_never_becomes_simulated_fill(tmp_path):
    manager = V14_3_MlegExecutionManager(
        trading_client=_RejectingBroker(),
        config=get_v14_3_config(),
        log_dir=str(tmp_path),
        paper_mode=True,
        allow_offline_simulation=False,
    )
    order = _entry(manager)
    assert manager.submit_order(order) is False
    assert order.state == OrderState.REJECTED
    assert "broker_error" in order.cancel_reason
    assert order.actual_fill_price == 0.0


def test_offline_simulation_requires_explicit_opt_in(tmp_path):
    manager = V14_3_MlegExecutionManager(
        trading_client=None,
        config=get_v14_3_config(),
        log_dir=str(tmp_path),
        paper_mode=True,
        allow_offline_simulation=False,
    )
    order = _entry(manager)
    assert manager.submit_order(order) is False
    assert order.state == OrderState.REJECTED

    sim = V14_3_MlegExecutionManager(
        trading_client=None,
        config=get_v14_3_config(),
        log_dir=str(tmp_path / "sim"),
        paper_mode=True,
        allow_offline_simulation=True,
    )
    simulated = _entry(sim)
    assert sim.submit_order(simulated) is True
    assert simulated.state == OrderState.FILLED


def test_mleg_parent_request_does_not_depend_on_top_level_side(tmp_path):
    broker = _CapturingBroker()
    manager = V14_3_MlegExecutionManager(
        trading_client=broker,
        config=get_v14_3_config(),
        log_dir=str(tmp_path),
        paper_mode=True,
    )
    order = _entry(manager)
    ok = manager.submit_order(order)
    # In an environment without the options-capable alpaca-py this fails closed;
    # with the required SDK installed it must serialize the current MLEG shape.
    if not ok and "sdk_not_available" in order.cancel_reason:
        pytest.skip("alpaca-py options models unavailable")
    assert ok
    assert broker.request is not None
    assert getattr(broker.request, "side", None) is None
    assert len(broker.request.legs) == 2
    assert float(broker.request.qty) == 1.0


def test_v14_3_symbol_size_is_actual_runner_state(tmp_path):
    fake_broker = _CapturingBroker()
    coin = V14_3_OptionsRunner(
        symbol="COIN",
        trading_client=fake_broker,
        option_data_client=_DummyOptionData(),
        config=get_v14_3_config(),
        paper_mode=True,
        log_dir=str(tmp_path / "coin"),
    )
    tsla = V14_3_OptionsRunner(
        symbol="TSLA",
        trading_client=fake_broker,
        option_data_client=_DummyOptionData(),
        config=get_v14_3_config(),
        paper_mode=True,
        log_dir=str(tmp_path / "tsla"),
    )
    assert coin.size_multiplier == 1.0
    assert tsla.size_multiplier == 0.5


def test_v14_3_watch_thresholds_are_used_not_v14_2_globals(tmp_path):
    runner = V14_3_OptionsRunner(
        symbol="COIN",
        trading_client=_CapturingBroker(),
        option_data_client=_DummyOptionData(),
        config=get_v14_3_config(),
        paper_mode=True,
        log_dir=str(tmp_path),
    )
    candidate = RouteCandidate(
        route="VWAP_PULLBACK",
        side="CALL",
        score=0.59,
        ic_spread=0.026,
        ev_over_debit=0.11,
        option_liquidity=0.70,
        expected_move=2.0,
    )
    assert runner._passes_v14_3_watch(candidate) is True
