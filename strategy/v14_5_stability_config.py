"""V14.5 stable options policy.

V14.5 keeps the V14.3 high-vol core/runner economics and adds a directional
hysteresis layer plus option-native contract quality metrics. The purpose is
not to make every short-lived CALL/PUT score change tradable. A side must earn
and retain directional permission, while an actual reversal must be stronger
and more persistent than an initial entry.

Status: PAPER/REPLAY VALIDATION CANDIDATE. No live-trading permission is changed.
"""

from __future__ import annotations

from strategy.v14_3_highvol_config import get_v14_3_config


V14_5_STABLE_OPTIONS = get_v14_3_config()
V14_5_STABLE_OPTIONS.update({
    "version": "14.5",
    "status": "STABLE_DIRECTION_OPTIONS_PAPER_VALIDATION",
    "profile_scope": "HIGH_VOL_LIQUID_OPTIONS_WITH_REVERSAL_HYSTERESIS",
    "live_trading_enabled": False,

    # Direction admission. Initial entries need brief consensus, not a 20-30m
    # warm-up. Existing bias receives hysteresis: the opposite side must be
    # materially stronger for several consecutive observations before it can flip.
    "direction_consensus_window": 4,
    "direction_consensus_required": 2,
    "direction_min_score_edge": 0.06,
    "direction_min_bias_score": 0.58,
    "reversal_min_route_score": 0.68,
    "reversal_min_score_edge": 0.12,
    "reversal_consecutive_bars": 3,
    "reversal_min_vwap_distance_atr": 0.10,
    "reversal_cooldown_bars": 4,
    "stop_reversal_cooldown_bars": 7,
    "target_reentry_cooldown_bars": 2,
    "block_opposite_while_open": True,
    "block_opposite_while_pending": True,
    "cancel_opposite_watches_on_bias_lock": True,

    # Exit hysteresis. Hard risk exits remain immediate. Only continuation-decay
    # exits require structure to remain broken for consecutive observations, with
    # a small ATR buffer around VWAP so one noisy cross cannot churn the spread.
    "exit_reversal_confirm_bars": 2,
    "exit_vwap_hysteresis_atr": 0.05,

    # Option-native quality. These augment, rather than replace, V14.3's
    # composite-spread, mid-inflation, reward/risk and move-consumed checks.
    "option_delta_target_abs": 0.55,
    "option_long_delta_min_abs": 0.40,
    "option_long_delta_max_abs": 0.72,
    "option_min_delta_spread_abs": 0.07,
    "option_max_theta_burden_pct_of_mid": 0.12,
    "option_max_leg_iv_skew": 0.15,
    "option_min_native_quality_score": 0.55,
    "option_missing_greeks_size_multiplier": 0.65,

    # Stability-aware sizing. A recently changed/weakly-established bias can
    # participate, but cannot receive the same debit allocation as a mature side.
    "new_bias_size_multiplier": 0.75,
    "recent_reversal_size_multiplier": 0.50,
    "stable_bias_full_size_after_bars": 4,
})


def get_v14_5_config() -> dict:
    """Return an isolated V14.5 config copy."""
    cfg = dict(V14_5_STABLE_OPTIONS)
    cfg["routes_allowed"] = list(V14_5_STABLE_OPTIONS["routes_allowed"])
    cfg["routes_soft_only"] = list(V14_5_STABLE_OPTIONS["routes_soft_only"])
    return cfg
