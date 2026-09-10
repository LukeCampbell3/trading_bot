"""V14.5 stability regression tests."""

from types import SimpleNamespace

from strategy.options_reversal_guard import OptionsReversalGuard
from strategy.route_conditioning import RouteCandidate
from strategy.spread_quality_gate import OptionLeg
from strategy.v14_5_options_runner import OptionNativeSpreadQualityGate
from strategy.v14_5_stability_config import get_v14_5_config


def _c(side: str, score: float, route: str = None):
    if route is None:
        route = "PUT_REJECTION" if side == "PUT" else "VWAP_PULLBACK"
    return RouteCandidate(
        route=route,
        side=side,
        score=score,
        ic_spread=0.04,
        ev_over_debit=0.20,
        option_liquidity=0.8,
        expected_move=2.0,
    )


def test_initial_bias_needs_short_consensus_not_long_warmup():
    cfg = get_v14_5_config()
    guard = OptionsReversalGuard(cfg)
    first = guard.select_candidates(
        "COIN", [_c("CALL", 0.70)], price=101, vwap=100, atr=1.0
    )
    assert first.allowed_candidates == []
    assert first.reason == "direction_consensus_building"
    second = guard.select_candidates(
        "COIN", [_c("CALL", 0.71)], price=101, vwap=100, atr=1.0
    )
    assert second.bias == "CALL"
    assert second.allowed_candidates
    assert second.bias_changed is True


def test_choppy_one_bar_opposite_signal_does_not_flip_bias():
    cfg = get_v14_5_config()
    guard = OptionsReversalGuard(cfg)
    for s in (0.70, 0.72):
        guard.select_candidates("COIN", [_c("CALL", s)], price=101, vwap=100, atr=1.0)
    assert guard.state("COIN").bias == "CALL"

    # A single strong PUT observation is insufficient for a reversal.
    d = guard.select_candidates(
        "COIN", [_c("CALL", 0.54), _c("PUT", 0.75)],
        price=99.7, vwap=100.0, atr=1.0,
    )
    assert guard.state("COIN").bias == "CALL"
    assert d.reason == "reversal_confirmation_building"
    assert all(c.side == "CALL" for c in d.allowed_candidates)


def test_persistent_strong_opposite_signal_can_reverse_when_flat():
    cfg = get_v14_5_config()
    guard = OptionsReversalGuard(cfg)
    for s in (0.70, 0.72):
        guard.select_candidates("COIN", [_c("CALL", s)], price=101, vwap=100, atr=1.0)
    assert guard.state("COIN").bias == "CALL"

    last = None
    for s in (0.74, 0.75, 0.76):
        last = guard.select_candidates(
            "COIN", [_c("CALL", 0.50), _c("PUT", s)],
            price=99.7, vwap=100.0, atr=1.0,
        )
    assert last is not None
    assert guard.state("COIN").bias == "PUT"
    assert last.bias_changed is True
    assert last.reason.startswith("confirmed_reversal")
    assert all(c.side == "PUT" for c in last.allowed_candidates)


def test_open_call_risk_hard_blocks_put_reversal():
    cfg = get_v14_5_config()
    guard = OptionsReversalGuard(cfg)
    guard.force_bias("COIN", "CALL")
    for _ in range(5):
        d = guard.select_candidates(
            "COIN", [_c("PUT", 0.95)],
            price=98, vwap=100, atr=1.0,
            open_risk_side="CALL",
        )
    assert guard.state("COIN").bias == "CALL"
    assert d.reason == "open_risk_direction_lock"
    assert d.allowed_candidates == []


def test_stop_exit_creates_longer_reversal_cooldown():
    cfg = get_v14_5_config()
    guard = OptionsReversalGuard(cfg)
    guard.force_bias("COIN", "CALL")
    guard.record_exit("COIN", "CALL", "package_stop")
    assert guard.state("COIN").cooldown_remaining == cfg["stop_reversal_cooldown_bars"]

    # Even a strong opposite signal cannot flip during the cooldown.
    for _ in range(3):
        guard.select_candidates(
            "COIN", [_c("CALL", 0.45), _c("PUT", 0.95)],
            price=98, vwap=100, atr=1.0,
        )
    assert guard.state("COIN").bias == "CALL"


def _leg(symbol, side, bid, ask, strike, delta, iv, theta):
    leg = OptionLeg(
        contract_symbol=symbol,
        side=side,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        strike=strike,
        dte=7,
        delta=delta,
        iv=iv,
    )
    leg.theta = theta
    leg.gamma = 0.01
    leg.vega = 0.10
    return leg


def test_option_native_gate_accepts_balanced_vertical():
    cfg = get_v14_5_config()
    # Isolate the option-native assertions from the legacy V14.3 quote-width cap.
    cfg["max_composite_spread_pct_of_mid"] = 0.50
    cfg["max_limit_chase_pct_of_debit"] = 0.50
    gate = OptionNativeSpreadQualityGate(cfg)
    long = _leg("C1", "buy", 2.00, 2.10, 100, 0.56, 0.55, -0.12)
    short = _leg("C2", "sell", 0.95, 1.05, 105, 0.34, 0.56, -0.07)
    r = gate.evaluate(
        long_leg=long,
        short_leg=short,
        underlying_price=101,
        underlying_price_at_watch=100.5,
        estimated_debit_at_watch=1.05,
        target_price=103.5,
        max_debit=300,
        route="VWAP_PULLBACK",
        iv_at_watch=0.55,
    )
    assert r.passed is True
    assert r.greeks_available is True
    assert r.delta_spread_abs > cfg["option_min_delta_spread_abs"]
    assert r.option_native_quality_score >= cfg["option_min_native_quality_score"]


def test_option_native_gate_rejects_bad_delta_structure():
    cfg = get_v14_5_config()
    cfg["max_composite_spread_pct_of_mid"] = 0.50
    cfg["max_limit_chase_pct_of_debit"] = 0.50
    gate = OptionNativeSpreadQualityGate(cfg)
    long = _leg("C1", "buy", 2.00, 2.10, 100, 0.84, 0.55, -0.12)
    short = _leg("C2", "sell", 0.95, 1.05, 105, 0.70, 0.56, -0.07)
    r = gate.evaluate(
        long_leg=long,
        short_leg=short,
        underlying_price=101,
        underlying_price_at_watch=100.5,
        estimated_debit_at_watch=1.05,
        target_price=103.5,
        max_debit=300,
        route="VWAP_PULLBACK",
        iv_at_watch=0.55,
    )
    assert r.passed is False
    assert r.rejection_reason.startswith("long_delta_out_of_band")
