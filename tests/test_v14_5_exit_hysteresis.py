"""Focused tests for V14.5 continuation-exit debounce."""

from types import SimpleNamespace

from strategy.v14_5_stable_options import V14_5_OptionsTrader
from strategy.v14_5_stability_config import get_v14_5_config
from strategy.watch_ticket import TicketSide


class _DummyBroker:
    def get_all_positions(self):
        return []


class _DummyData:
    pass


def _trader(tmp_path):
    return V14_5_OptionsTrader(
        symbol="COIN",
        trading_client=_DummyBroker(),
        option_data_client=_DummyData(),
        config=get_v14_5_config(),
        paper_mode=True,
        log_dir=str(tmp_path),
        allow_offline_simulation=True,
    )


def test_call_exit_requires_atr_buffer(tmp_path):
    t = _trader(tmp_path)
    t._latest_atr = 2.0
    ticket = SimpleNamespace(side=TicketSide.CALL)
    # 0.02 below VWAP is inside the configured 0.10 (= .05 ATR) buffer.
    assert t._continuation_failed(ticket, 99.98, 100.00) is False
    assert t._continuation_failed(ticket, 99.89, 100.00) is True


def test_put_exit_requires_atr_buffer(tmp_path):
    t = _trader(tmp_path)
    t._latest_atr = 2.0
    ticket = SimpleNamespace(side=TicketSide.PUT)
    assert t._continuation_failed(ticket, 100.02, 100.00) is False
    assert t._continuation_failed(ticket, 100.11, 100.00) is True


def test_v14_5_defaults_keep_live_disabled():
    cfg = get_v14_5_config()
    assert cfg["live_trading_enabled"] is False
    assert cfg["exit_reversal_confirm_bars"] >= 2
    assert cfg["reversal_consecutive_bars"] >= 3
    assert cfg["reversal_min_score_edge"] > cfg["direction_min_score_edge"]
