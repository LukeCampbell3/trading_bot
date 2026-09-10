from datetime import datetime, timezone

from political_signals.pelosi_tail import (
    PelosiDisclosure,
    PelosiTailPolicy,
    PelosiTailPoller,
)


class FakeClient:
    def __init__(self, rows):
        self.rows = rows

    def fetch_recent(self):
        return list(self.rows)


def disclosure(**overrides):
    raw = {
        "Representative": "Nancy Pelosi",
        "ReportDate": "2026-09-10",
        "TransactionDate": "2026-09-03",
        "Ticker": "NVDA",
        "Transaction": "Purchase",
        "Range": "$250,001 - $500,000",
        "House": "House",
        "TickerType": "Stock",
        "Description": "NVIDIA CORPORATION COMMON STOCK",
    }
    raw.update(overrides)
    return PelosiDisclosure.from_quiver(raw)


def test_filters_and_parses_pelosi_trade():
    d = disclosure()
    assert d.is_pelosi is True
    assert d.ticker == "NVDA"
    assert d.disclosure_lag_days == 7
    assert d.disclosed_direction == "BULLISH"
    assert d.amount_bounds == (250001.0, 500000.0)


def test_bought_put_is_bearish_exit_only():
    d = disclosure(
        TickerType="Options",
        Description="Purchased put options",
    )
    decision = PelosiTailPolicy().decide(d)
    assert d.disclosed_direction == "BEARISH"
    assert decision.action == "EXIT_ONLY"


def test_ambiguous_option_is_never_auto_tailed():
    d = disclosure(TickerType="Options", Description="Option position")
    decision = PelosiTailPolicy().decide(d)
    assert d.disclosed_direction == "UNKNOWN"
    assert decision.action == "WATCH"


def test_old_disclosure_is_rejected():
    d = disclosure(TransactionDate="2026-07-01")
    decision = PelosiTailPolicy(max_disclosure_lag_days=45).decide(d)
    assert decision.action == "IGNORE"
    assert "disclosure_too_old" in decision.reasons


def test_large_runup_does_not_chase_delayed_buy():
    d = disclosure()
    decision = PelosiTailPolicy(max_buy_runup_since_trade=.25).decide(
        d, trade_to_now_return=.40
    )
    assert decision.action == "WATCH"
    assert decision.target_notional_pct == 0
    assert "positive_move_already_consumed" in decision.reasons


def test_fresh_large_purchase_creates_bounded_buy_signal():
    d = disclosure()
    decision = PelosiTailPolicy(base_max_notional_pct=.08).decide(
        d,
        first_seen_at=datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc),
        trade_to_now_return=.02,
    )
    assert decision.action == "BUY"
    assert 0 < decision.target_notional_pct <= .08
    assert decision.disclosure_lag_days == 7


def test_first_poll_seeds_existing_without_backlog_trade(tmp_path):
    d = disclosure()
    client = FakeClient([d])
    poller = PelosiTailPoller(
        client,
        state_path=str(tmp_path / "state.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
        seed_existing_on_first_run=True,
    )
    assert poller.poll_once() == []
    assert d.fingerprint in poller.state["seen"]


def test_only_new_disclosure_emits_after_baseline(tmp_path):
    first = disclosure(Ticker="NVDA")
    second = disclosure(Ticker="AAPL", TransactionDate="2026-09-05")
    client = FakeClient([first])
    poller = PelosiTailPoller(
        client,
        state_path=str(tmp_path / "state.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
        seed_existing_on_first_run=True,
    )
    assert poller.poll_once() == []
    client.rows = [first, second]
    decisions = poller.poll_once()
    assert len(decisions) == 1
    assert decisions[0].ticker == "AAPL"


def test_sale_signal_never_opens_naked_short():
    d = disclosure(Transaction="Sale")
    decision = PelosiTailPolicy().decide(d)
    assert decision.action == "EXIT_ONLY"
    assert decision.target_notional_pct == 0
