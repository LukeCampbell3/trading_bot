"""V14.5.1 validation-hardening policy.

V14.5.1 keeps V14.5's directional hysteresis and core/runner economics while
hardening the parts that were not yet safe enough for broker-backed validation:

* dimensionless trend inputs;
* unique-market-bar direction state;
* empirical, independently labelled option edge instead of route-score-derived IC/EV;
* live route-decay confirmation;
* explicit missing-Greeks sizing haircut;
* partial-fill reconciliation;
* bounded stale hard-exit cancel/reprice;
* proof-stage debit and kill-switch limits.

Live trading stays disabled.  The policy is intended for replay and Alpaca paper
validation until fill-adjusted evidence passes the promotion gates.
"""

from __future__ import annotations

from strategy.v14_5_stability_config import get_v14_5_config


V14_5_1_VALIDATION_HARDENED = get_v14_5_config()
V14_5_1_VALIDATION_HARDENED.update({
    "version": "14.5.1",
    "status": "VALIDATION_HARDENED_PAPER_REPLAY",
    "profile_scope": "HIGH_VOL_LIQUID_OPTIONS_VALIDATION_HARDENED",
    "live_trading_enabled": False,

    # Trend is fractional price slope per one-minute bar, not dollars/minute.
    "trend_units": "fraction_per_minute",

    # Reversal persistence advances on unique market bars only.
    "direction_unique_bar_only": True,

    # A route must still look like the route that created the watch.  This closes
    # the old route_score_now=ticket.route_score loophole.
    "route_decay_floor_ratio": 0.90,

    # Empirical edge is learned from timestamp-safe option-spread outcomes.  Until
    # enough independent labels exist the strategy may gather PAPER/REPLAY evidence
    # at reduced size, but it does not pretend route-score transforms are IC/EV.
    "empirical_edge_min_samples": 20,
    "empirical_edge_window": 200,
    "empirical_edge_prior_strength": 20.0,
    "empirical_min_ic_spread": 0.02,
    "empirical_min_ev_over_debit": 0.03,
    "empirical_unvalidated_size_multiplier": 0.50,
    "empirical_negative_edge_blocks": True,

    # Missing option-native information is allowed in paper/replay so evidence can
    # accumulate, but gets the intended haircut.  Live remains disabled regardless.
    "option_missing_greeks_size_multiplier": 0.65,

    # Partial MLEG strategy-unit fills are reconciled instead of being mistaken for
    # a missed fill.  Cancel the unfilled remainder quickly after exposure exists.
    "partial_fill_cancel_after_seconds": 5.0,

    # Hard exit recovery.  Natural-bid limit is refreshed quickly if the market
    # moves.  Targets are not chased; stops/locks are.
    "hard_exit_requote_seconds": 3.0,
    "soft_exit_requote_seconds": 15.0,
    "hard_exit_max_requotes": 4,
    "hard_exit_reprice_step_pct": 0.015,
    "hard_exit_max_concession_pct": 0.12,

    # Proof-stage risk.  V14.3's 35% debit exposure was too large for a $1k account
    # before real fill quality is established.
    "max_open_debit_exposure_pct": 0.18,
    "daily_kill_loss_pct": -0.05,
    "weekly_kill_loss_pct": -0.10,

    # Multi-vertical search.  Search broadly but cap API/CPU work.
    "vertical_selector_max_pairs": 24,
    "vertical_selector_max_expirations": 3,
    "vertical_selector_min_width": 2.5,
    "vertical_selector_max_width": 10.0,
    "vertical_selector_min_score": 0.48,
})


def get_v14_5_1_config() -> dict:
    cfg = dict(V14_5_1_VALIDATION_HARDENED)
    cfg["routes_allowed"] = list(V14_5_1_VALIDATION_HARDENED["routes_allowed"])
    cfg["routes_soft_only"] = list(V14_5_1_VALIDATION_HARDENED["routes_soft_only"])
    return cfg
