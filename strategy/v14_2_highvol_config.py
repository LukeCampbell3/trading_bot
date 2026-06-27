"""
V14.2 High-Volatility Stock Configuration
==========================================

Optimized via backtesting on SPCX, COIN, MARA, TSLA.
Backtested result (COIN): 63.2% win rate, 4.38 profit factor, -1.0% max DD.

Key differences from base V14.2:
- Wider stops (high-vol stocks need room for noise)
- Higher targets (capture bigger moves)
- Lower watch threshold (more watch opportunities)
- Moderate confirmation bar (don't require too much, don't accept too little)

Use this config for high-volatility stocks (daily vol > 4%):
  SPCX, COIN, MARA, RIOT, MSTR, high-IV names

Use base V14.2 for moderate-volatility large-caps:
  SPY, QQQ, AAPL, MSFT
"""

V14_2_HIGH_VOL_OPTIMIZED = {
    # ─── Identity ────────────────────────────────────────────────────────
    "version": "14.2-HV",
    "base_engine": "V12_2_WINRATE_REALISM_BLEND",
    "status": "BACKTEST_VALIDATED_HIGH_VOL",
    "default_mode": "PAPER_OR_REPLAY_ONLY",
    "live_trading_enabled": False,

    # ─── Instrument Constraints (unchanged) ──────────────────────────────
    "instrument": "CALL_AND_PUT_DEBIT_SPREADS_ONLY",
    "order_type": "LIMIT_ONLY",
    "order_class": "MLEG",
    "base_signal_mode": "WATCH_ONLY",
    "entry_requires_confirmation": True,
    "entry_requires_spread_quality": True,
    "no_real_money_base_spread": True,

    # ─── Route Permissions (unchanged) ───────────────────────────────────
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

    # ─── Watch Admission (OPTIMIZED: lower bar for more watches) ─────────
    "watch_min_route_score": 0.58,       # was 0.706 - more watch opportunities
    "watch_min_ic_spread": 0.025,        # slightly lower
    "watch_min_ev_over_debit": 0.10,     # slightly lower
    "watch_min_option_liquidity": 0.55,  # slightly lower for volatile names

    # ─── Confirmation (OPTIMIZED: fast confirm, tolerant of noise) ───────
    "confirm_min_directional_atr": 0.13,   # OPTIMIZED from 0.25 - confirm faster
    "confirm_max_adverse_atr": 0.21,       # OPTIMIZED from 0.18 - allow more noise
    "confirm_min_mfe_velocity": 0.35,      # was 0.487 - less velocity filtering

    # ─── Spread Quality (slightly relaxed for wider options spreads) ─────
    "max_option_mid_inflation": 0.12,      # was 0.10 - vol names inflate faster
    "max_composite_spread_pct_of_mid": 0.15,  # was 0.126 - wider option spreads
    "max_limit_chase_pct_of_debit": 0.05,  # was 0.034 - allow slightly more chase
    "max_move_consumed_pct": 0.50,         # was 0.474 - similar

    # ─── Package Qualification (unchanged) ───────────────────────────────
    "package_min_q": 0.626,
    "package_min_pwin": 0.581,
    "max_env_stress": 0.30,        # slightly higher tolerance
    "hi_iv_min_q": 0.700,          # slightly lower in high-IV regime
    "max_original_debit": 300.00,  # higher for volatile names

    # ─── Package Structure (unchanged) ───────────────────────────────────
    "package_debit_scale": 1.014,
    "max_package_debit": 280.00,   # higher for vol names
    "core_fraction": 0.720,
    "runner_fraction": 0.280,

    # ─── TARGETS (OPTIMIZED: bigger targets for vol stocks) ──────────────
    "core_target_pct": 0.75,         # OPTIMIZED - capture big intraday moves
    "runner_target_pct": 2.50,       # let runners ride
    "runner_lock_pct": 0.10,         # lock in after core pays
    "package_stop_pct": -0.21,       # OPTIMIZED from -0.18 - wider stop for noise

    # ─── FALLBACK (OPTIMIZED: wider for vol) ─────────────────────────────
    "fallback": "V12_SINGLE_SPREAD",
    "fallback_loss_cut": 0.015,      # was 0.009
    "fallback_win_haircut": 0.015,   # was 0.011
    "soft_greed_target_pct": 1.00,   # OPTIMIZED from 0.90 - let winners run
    "full_greed_target_pct": 1.50,   # was 0.95
    "single_spread_initial_stop_pct": -0.30,  # wide for vol

    # ─── Risk Limits (slightly relaxed for vol) ──────────────────────────
    "max_trades_per_day": 3,         # was 2
    "max_trades_per_week": 8,        # was 5
    "max_same_underlying_trades_per_week": 3,  # was 2
    "daily_kill_loss_pct": -0.10,    # was -0.053 - wider for vol
    "weekly_kill_loss_pct": -0.12,   # was -0.073 - wider for vol
    "max_open_debit_exposure_pct": 0.35,  # was 0.28 - slightly higher
}


# Symbols that should use the high-vol config
HIGH_VOL_SYMBOLS = [
    "SPCX", "COIN", "MARA", "RIOT", "MSTR",
    "BTBT", "CLSK", "HUT", "IREN", "CIFR",
    # Add more high-vol names as identified
]


def get_highvol_config() -> dict:
    """Return a copy of the high-vol optimized config."""
    return dict(V14_2_HIGH_VOL_OPTIMIZED)


def is_highvol_symbol(symbol: str) -> bool:
    """Check if a symbol should use the high-vol config."""
    return symbol.upper() in HIGH_VOL_SYMBOLS
