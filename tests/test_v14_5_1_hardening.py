"""Regression tests for the V14.5.1 validation-hardening pass."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from execution.mleg_execution_manager import MlegLeg, MlegOrder, OrderState
from execution.v14_5_1_mleg_execution_manager import V14_5_1_MlegExecutionManager
from strategy.empirical_option_edge import EmpiricalOptionEdgeMemory
from strategy.route_conditioning import RouteCandidate
from strategy.spread_quality_gate import OptionLeg
from strategy.v14_5_1_config import get_v14_5_1_config
from strategy.v14_5_1_features import NormalizedSessionWarmFeatureEngine
from strategy.v14_5_1_hardened_options import (
    BarAwareOptionsReversalGuard,
    LiveRouteConfirmationEngine,
    V14_5_1_OptionsTrader,
)
from strategy.v14_5_1_vertical_selector import RankedVerticalSelector
from strategy.watch_ticket import WatchTicket, TicketSide, TicketStatus


def _candidate(side="CALL", score=0.70, route="VWAP_PULLBACK"):
    return RouteCandidate(
        route=route,
        side=side,
        score=score,
        option_liquidity=0.8,
        expected_move=2.0,
    )


def _frame(price=300.0):
    idx = []
    rows = []
    prior = pd.Timestamp("2026-09-09 15:00", tz="America/New_York")
    for i in range(30):
        px = price * (0.995 + i * 0.00005)
        idx.append((prior + pd.Timedelta(minutes=i)).tz_convert("UTC"))
        rows.append((px, px + .1, px - .1, px, 100.0))
    cur = pd.Timestamp("2026-09-10 09:30", tz="America/New_York")
    for i in range(20):
        px = price + i * 0.05
        idx.append((cur + pd.Timedelta(minutes=i)).tz_convert("UTC"))
        rows.append((px, px + .1, px - .1, px, 100.0))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["open", "high", "low", "close", "volume"])


def test_trend_is_dimensionless_fraction_per_minute():
    out = NormalizedSessionWarmFeatureEngine(min_session_bars=5).compute(_frame(367.0))
    raw = out["trend_slope_raw_dollars_per_min"]
    assert out["trend_units"] == "fraction_per_minute"
    assert abs(out["trend_slope"] - raw / out["price"]) < 1e-12
    assert abs(out["trend_slope"]) < abs(raw)


def test_duplicate_poll_does_not_advance_direction_consensus():
    cfg = get_v14_5_1_config()
    guard = BarAwareOptionsReversalGuard(cfg)
    first = guard.select_candidates(
        "COIN", [_candidate(score=.70)], price=101, vwap=100, atr=1.0,
        observation_id="2026-09-10T13:30:00Z",
    )
    assert first.reason == "direction_consensus_building"
    duplicate = guard.select_candidates(
        "COIN", [_candidate(score=.72)], price=101, vwap=100, atr=1.0,
        observation_id="2026-09-10T13:30:00Z",
    )
    assert duplicate.reason == "duplicate_market_bar_ignored"
    assert guard.state("COIN").bias == "NEUTRAL"
    next_bar = guard.select_candidates(
        "COIN", [_candidate(score=.73)], price=101, vwap=100, atr=1.0,
        observation_id="2026-09-10T13:31:00Z",
    )
    assert next_bar.bias == "CALL"


def test_empirical_edge_is_independent_of_route_score_transform(tmp_path):
    mem = EmpiricalOptionEdgeMemory(
        min_samples=4, window=20, prior_strength=0,
        log_path=str(tmp_path / "edge.csv"),
    )
    for score, ret in [(0.60, -0.10), (0.65, 0.02), (0.72, 0.15), (0.80, 0.30)]:
        mem.record(symbol="COIN", route="VWAP_PULLBACK", route_score=score, spread_return=ret)
    st = mem.stats("COIN", "VWAP_PULLBACK")
    assert st.ready is True
    assert st.ic_spread > 0.5
    assert st.ev_over_debit > 0
    c = _candidate(score=.74)
    mem.annotate_candidate(c, "COIN")
    assert c.ic_spread == st.ic_spread
    assert c.ev_over_debit == st.ev_over_debit
    assert c.ic_spread != c.score * 0.05


def _watch_ticket():
    return WatchTicket(
        symbol="COIN",
        route="VWAP_PULLBACK",
        side=TicketSide.CALL,
        timestamp_created=datetime.utcnow(),
        underlying_price_at_watch=100.0,
        vwap_at_watch=100.0,
        atr_at_watch=1.0,
        route_score=0.70,
        ic_spread=0.04,
        expected_ev_over_debit=0.20,
        option_liquidity_score=0.8,
        expected_move_to_target=2.0,
        estimated_debit_at_watch=1.0,
    )


def test_confirmation_rejects_live_route_decay_even_when_watch_score_was_good():
    cfg = get_v14_5_1_config()
    owner = SimpleNamespace(_latest_route_scores={("CALL", "VWAP_PULLBACK"): 0.59})
    engine = LiveRouteConfirmationEngine(owner, cfg)
    ticket = _watch_ticket()
    result = engine.check_confirmation(
        ticket=ticket,
        current_price=100.5,
        current_vwap=100.0,
        current_atr=1.0,
        high_since_watch=100.5,
        low_since_watch=99.95,
        mfe_velocity=0.6,
        env_stress=0.0,
        route_score_now=ticket.route_score,
    )
    assert result.confirmed is False
    assert result.reason.startswith("route_decayed_live")


class _BrokerOrder:
    status = "partially_filled"
    filled_qty = "1"
    qty = "3"
    filled_avg_price = "1.27"


class _PartialBroker:
    def get_order_by_id(self, order_id):
        return _BrokerOrder()

    def get_all_positions(self):
        return []


def test_execution_manager_captures_partial_strategy_units(tmp_path):
    mgr = V14_5_1_MlegExecutionManager(
        trading_client=_PartialBroker(), config=get_v14_5_1_config(),
        log_dir=str(tmp_path), paper_mode=True,
    )
    order = MlegOrder(direction="OPEN", quantity=3, broker_order_id="abc", composite_mid=1.20)
    state = mgr.check_order_status(order)
    assert state == OrderState.PARTIALLY_FILLED
    assert order.broker_filled_qty == 1
    assert order.broker_remaining_qty == 2
    assert order.broker_filled_avg_price == 1.27


class _DummyBroker:
    def get_all_positions(self):
        return []


class _DummyData:
    pass


def _trader(tmp_path):
    return V14_5_1_OptionsTrader(
        symbol="COIN", trading_client=_DummyBroker(), option_data_client=_DummyData(),
        config=get_v14_5_1_config(), paper_mode=True, log_dir=str(tmp_path),
        allow_offline_simulation=True,
    )


def test_partial_entry_becomes_managed_fallback_position(tmp_path):
    t = _trader(tmp_path)
    ticket = _watch_ticket()
    ticket.status = TicketStatus.CONFIRMED
    t.ticket_book.active_tickets[ticket.ticket_id] = ticket
    order = MlegOrder(direction="OPEN", quantity=3, intended_limit=1.25, actual_fill_price=1.27)
    order.broker_filled_avg_price = 1.27
    package = SimpleNamespace(is_package=True)
    contract = SimpleNamespace(symbol="COINX", option_type="call", strike=100.0, expiration="2026-09-18")
    short = SimpleNamespace(symbol="COINY", option_type="call", strike=105.0, expiration="2026-09-18")
    t._pending_entry[ticket.ticket_id] = {
        "order": order, "package": package, "qty": 3,
        "long_contract": contract, "short_contract": short,
    }
    t._sync_partial_entry(ticket.ticket_id, 1)
    plan = t._positions[ticket.ticket_id]
    assert plan.total_qty == 1
    assert plan.mode == "FALLBACK"
    assert plan.fallback_qty == 1
    assert t._pending_entry[ticket.ticket_id]["activated_qty"] == 1


def _leg(symbol, bid, ask, strike, delta, iv):
    leg = OptionLeg(
        contract_symbol=symbol, side="buy", bid=bid, ask=ask,
        mid=(bid + ask) / 2, strike=strike, dte=7, delta=delta, iv=iv,
    )
    leg.theta = -0.10
    return leg


def test_ranked_selector_prefers_tighter_delta_fit_structure():
    cfg = get_v14_5_1_config()
    selector = object.__new__(RankedVerticalSelector)
    selector.cfg = cfg
    good_l = _leg("L1", 5.00, 5.05, 100, .55, .40)
    good_s = _leg("S1", 2.95, 3.00, 105, .35, .405)
    bad_l = _leg("L2", 5.00, 5.50, 100, .78, .40)
    bad_s = _leg("S2", 2.50, 3.00, 105, .62, .45)
    lc = SimpleNamespace(strike=100.0)
    sc = SimpleNamespace(strike=105.0)
    good = selector._score_pair(good_l, good_s, 101.0, lc, sc)
    bad = selector._score_pair(bad_l, bad_s, 101.0, lc, sc)
    assert good > bad


def test_v14_5_1_proof_stage_risk_is_reduced():
    cfg = get_v14_5_1_config()
    assert cfg["live_trading_enabled"] is False
    assert cfg["max_open_debit_exposure_pct"] <= 0.20
    assert cfg["daily_kill_loss_pct"] >= -0.05
    assert cfg["weekly_kill_loss_pct"] >= -0.10
