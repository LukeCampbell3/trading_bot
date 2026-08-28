"""
V14.2 Core Runner Replacement - Active Policy Configuration
Status: SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN

This config defines the active paper/simulation strategy parameters.
Live trading is DISABLED by default and requires explicit override.
"""

V14_2_CORE_RUNNER_REPLACEMENT = {
    # ─── Identity ────────────────────────────────────────────────────────
    "version": "14.2",
    "base_engine": "V12_2_WINRATE_REALISM_BLEND",
    "status": "SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN",
    "default_mode": "PAPER_OR_REPLAY_ONLY",
    "live_trading_enabled": False,

    # ─── Instrument Constraints ──────────────────────────────────────────
    "instrument": "CALL_AND_PUT_DEBIT_SPREADS_ONLY",
    "order_type": "LIMIT_ONLY",
    "order_class": "MLEG",
    "base_signal_mode": "WATCH_ONLY",

    # ─── Entry Requirements ──────────────────────────────────────────────
    "entry_requires_confirmation": True,
    "entry_requires_spread_quality": True,
    "no_real_money_base_spread": True,

    # ─── Route Permissions ───────────────────────────────────────────────
    "routes_allowed": [
        "VWAP_PULLBACK",
        "PULLBACK_CONTINUATION",
        "DOMINANT_TREND_PULLBACK",
        "PUT_REJECTION",
    ],
    "routes_soft_only": [
        "RAW_BREAKOUT",
        "LATE_BREAKOUT",
        "NEWS_SPIKE",
    ],

    # ─── Watch Admission Gates ───────────────────────────────────────────
    "watch_min_route_score": 0.706,
    "watch_min_ic_spread": 0.028,
    "watch_min_ev_over_debit": 0.127,
    "watch_min_option_liquidity": 0.623,

    # ─── Confirmation Gates ──────────────────────────────────────────────
    "confirm_min_directional_atr": 0.263,
    "confirm_max_adverse_atr": 0.151,
    "confirm_min_mfe_velocity": 0.487,

    # ─── Spread Quality Gates ────────────────────────────────────────────
    "max_option_mid_inflation": 0.10,
    "max_composite_spread_pct_of_mid": 0.126,
    "max_limit_chase_pct_of_debit": 0.034,
    "max_move_consumed_pct": 0.474,

    # ─── Package Qualification ───────────────────────────────────────────
    "package_min_q": 0.626,
    "package_min_pwin": 0.581,
    "max_env_stress": 0.252,
    "hi_iv_min_q": 0.750,
    "max_original_debit": 245.70,

    # ─── Package Structure ───────────────────────────────────────────────
    "package_debit_scale": 1.014,
    "max_package_debit": 229.36,
    "core_fraction": 0.720,
    "runner_fraction": 0.280,

    # ─── Package Targets & Stops ─────────────────────────────────────────
    "core_target_pct": 0.372,
    "runner_target_pct": 1.155,
    "runner_lock_pct": 0.066,
    "package_stop_pct": -0.081,

    # ─── V12.2 Fallback Parameters ──────────────────────────────────────
    "fallback": "V12_SINGLE_SPREAD",
    "fallback_loss_cut": 0.009,
    "fallback_win_haircut": 0.011,

    # ─── Single Spread (Greed) Targets ───────────────────────────────────
    "soft_greed_target_pct": 0.55,
    "full_greed_target_pct": 0.95,
    "single_spread_initial_stop_pct": -0.16,

    # ─── Risk Limits ─────────────────────────────────────────────────────
    # NOTE: max_trades_per_day/week are no longer hard blockers on their own.
    # RiskManager now derives how many trades the account can still afford
    # from capital headroom under max_open_debit_exposure_pct (dynamic,
    # opportunity-responsive). These two values now serve only as the
    # *soft baseline* the dynamic calc starts from before capital headroom
    # and the absolute circuit-breaker ceilings below bound it.
    "max_trades_per_day": 2,
    "max_trades_per_week": 5,
    "max_same_underlying_trades_per_week": 2,
    "daily_kill_loss_pct": -0.053,
    "weekly_kill_loss_pct": -0.073,
    "max_open_debit_exposure_pct": 0.28,

    # ─── Dynamic Gate Control ──────────────────────────────────────────────
    # Capital protection stays hard (kill switches + exposure cap above are
    # never loosened). Everything below adapts within bounded ranges so the
    # bot can take more of the *good* setups it finds instead of stopping
    # dead at an arbitrary trade count, while a route/environment on a cold
    # streak gets harder to enter (not permanently frozen).
    "dynamic_gates_enabled": True,

    # Circuit breakers: even with abundant capital headroom, cadence can
    # never exceed these — guards against a data glitch or feedback loop
    # spamming orders. Set comfortably above the old static caps so capital
    # headroom (not an arbitrary count) is normally the binding constraint.
    "absolute_max_trades_per_day_ceiling": 8,
    "absolute_max_trades_per_week_ceiling": 20,

    # How far a route's entry thresholds may adapt from their base value.
    # 0.85 = proven route can require 15% less to enter; 1.25 = a cold route
    # needs 25% more before it's admitted. Spread-quality/execution-risk
    # checks are never scaled — only opportunity-selection thresholds are.
    "gate_multiplier_min": 0.85,
    "gate_multiplier_max": 1.25,

    # A route/environment that would previously have been permanently
    # DISABLED/BLOCKED instead goes on cooldown, then gets a small number of
    # size-reduced "probation" trades to re-earn full trust with fresh data.
    "route_probation_cooldown_hours": 24,
    "route_probation_trade_limit": 1,
    "route_probation_size_multiplier": 0.5,
    "env_probation_cooldown_hours": 12,
    "env_probation_trade_limit": 1,
    "env_probation_size_multiplier": 0.5,
}


def get_config() -> dict:
    """Return a copy of the active V14.2 config."""
    return dict(V14_2_CORE_RUNNER_REPLACEMENT)
