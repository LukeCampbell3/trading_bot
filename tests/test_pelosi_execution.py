from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from political_signals.pelosi_execution import PelosiAlpacaExecutor
from political_signals.pelosi_tail import PelosiTailDecision


class FakeOrder:
    def __init__(self, oid, *, status="filled", filled_qty=0.0, filled_avg_price=100.0):
        self.id = oid
        self.status = status
        self.filled_qty = str(filled_qty)
        self.filled_avg_price = str(filled_avg_price)
        self.filled_at = datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc)


class FakeTrading:
    def __init__(self, *, market_open=True, equity=1000.0, buying_power=1000.0):
        self.market_open = market_open
        self.equity = equity
        self.buying_power = buying_power
        self.orders = {}
        self.submitted = []
        self.canceled = []
        self.counter = 0

    def get_account(self):
        return SimpleNamespace(status="ACTIVE", equity=str(self.equity), buying_power=str(self.buying_power))

    def get_clock(self):
        return SimpleNamespace(is_open=self.market_open)

    @staticmethod
    def _side(req):
        value = getattr(req.side, "value", req.side)
        return str(value).lower()

    def submit_order(self, order_data):
        self.counter += 1
        oid = f"order-{self.counter}"
        side = self._side(order_data)
        if side == "buy":
            notional = float(order_data.notional)
            qty = notional / 100.0
        else:
            qty = float(order_data.qty)
        order = FakeOrder(oid, filled_qty=qty, filled_avg_price=100.0)
        self.orders[oid] = order
        self.submitted.append(order_data)
        return order

    def get_order_by_id(self, oid):
        return self.orders[oid]

    def cancel_order_by_id(self, oid):
        self.canceled.append(oid)
        self.orders[oid].status = "canceled"


def decision(*, did="abc123", ticker="NVDA", action="BUY", pct=0.06):
    return PelosiTailDecision(
        disclosure_id=did,
        ticker=ticker,
        action=action,
        score=0.75,
        target_notional_pct=pct,
        disclosure_lag_days=7,
        amount_low=250001,
        amount_high=500000,
        first_seen_at="2026-09-10T15:00:00+00:00",
        reasons=["test"],
        estimated_trade_to_now_return=0.02,
    )


def executor(tmp_path, broker, **kwargs):
    return PelosiAlpacaExecutor(
        mode="paper",
        trading_client=broker,
        state_path=str(tmp_path / "exec.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
        **kwargs,
    )


def test_shadow_never_writes_broker(tmp_path):
    broker = FakeTrading()
    ex = PelosiAlpacaExecutor(
        mode="shadow",
        trading_client=broker,
        state_path=str(tmp_path / "exec.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
    )
    result = ex.process(decision())
    assert result.execution_action == "SHADOW_ONLY"
    assert broker.submitted == []


def test_buy_signal_places_real_paper_order_and_owns_fill(tmp_path):
    broker = FakeTrading()
    ex = executor(tmp_path, broker)
    result = ex.process(decision(pct=.06))
    assert result.execution_action == "BUY_SUBMITTED"
    assert result.requested_notional == 60.0
    assert len(broker.submitted) == 1
    assert ex.owned_qty("NVDA") == pytest.approx(.60)


def test_same_disclosure_is_idempotent_across_restart(tmp_path):
    broker = FakeTrading()
    ex = executor(tmp_path, broker)
    first = ex.process(decision())
    assert first.execution_action == "BUY_SUBMITTED"

    restarted = executor(tmp_path, broker)
    second = restarted.process(decision())
    assert second.execution_action == "DUPLICATE_IGNORED"
    assert len(broker.submitted) == 1


def test_market_closed_signal_queues_then_executes_at_open(tmp_path):
    broker = FakeTrading(market_open=False)
    ex = executor(tmp_path, broker)
    queued = ex.process(decision())
    assert queued.execution_action == "QUEUED"
    assert broker.submitted == []

    broker.market_open = True
    results = ex.process_pending()
    assert len(results) == 1
    assert results[0].execution_action == "BUY_SUBMITTED"
    assert len(broker.submitted) == 1


def test_risk_caps_override_oversized_signal(tmp_path):
    broker = FakeTrading(equity=1000, buying_power=1000)
    ex = executor(
        tmp_path, broker,
        max_order_notional_pct=.08,
        max_symbol_equity_pct=.10,
        max_strategy_equity_pct=.20,
    )
    result = ex.process(decision(pct=.50))
    assert result.requested_notional == 80.0


def test_bearish_signal_exits_only_strategy_owned_quantity(tmp_path):
    broker = FakeTrading()
    ex = executor(tmp_path, broker)
    ex.process(decision(did="buy-1", pct=.06))
    assert ex.owned_qty("NVDA") == pytest.approx(.60)

    result = ex.process(decision(did="sell-1", action="EXIT_ONLY", pct=0.0))
    assert result.execution_action == "EXIT_SUBMITTED"
    sell_req = broker.submitted[-1]
    assert float(sell_req.qty) == pytest.approx(.60)
    assert ex.owned_qty("NVDA") == pytest.approx(0.0)


def test_exit_does_nothing_without_pelosi_owned_position(tmp_path):
    broker = FakeTrading()
    ex = executor(tmp_path, broker)
    result = ex.process(decision(did="sell-2", action="EXIT_ONLY", pct=0.0))
    assert result.execution_action == "NO_ACTION"
    assert result.reason == "no_pelosi_owned_position"
    assert broker.submitted == []


def test_live_mode_requires_explicit_second_gate(tmp_path, monkeypatch):
    broker = FakeTrading()
    monkeypatch.delenv("PELOSI_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="Live Pelosi execution is locked"):
        PelosiAlpacaExecutor(
            mode="live",
            trading_client=broker,
            state_path=str(tmp_path / "exec.json"),
            audit_path=str(tmp_path / "audit.jsonl"),
        )
