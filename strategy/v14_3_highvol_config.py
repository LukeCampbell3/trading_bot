"""
V14.3 High-Vol Core Runner - Local Optimum
===========================================
Status: LOCAL_OPTIMUM_REPLAY_VALIDATED_NOT_LIVE_PROVEN

This is NOT a global optimum. It is a symbol-regime-specific profile for
high-volatility liquid option names where:
- Early confirmation beats late certainty
- Wide stops prevent chop-out on noise
- Large targets capture the outsized moves these names produce

Backtest validation:
  COIN: 75% win rate, 6.10 PF, +$698 P&L  -> PROMOTE
  TSLA: 35.7% win rate, 2.19 PF, +$458    -> PROBATION (half-size)
  MARA: 41.7% win rate, 0.78 PF, -$4      -> OBSERVE ONLY
  SPCX: no option chain                    -> DISABLED

DO NOT apply globally. Use only for symbols in allowed_symbols list.
The base V14.2 config remains active for moderate-vol large-caps.

Paper fill caveat: Alpaca paper fills are simulated and do not capture
live market impact, latency slippage, queue position, or regulatory fees.
This matters especially for option spreads.
"""

from __future__ import annotations
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════
# V14.3 HIGH-VOL PROFILE
# ═══════════════════════════════════════════════════════════════════════════

V14_3_HIGH_VOL_CORE_RUNNER = {
    # ─── Identity ────────────────────────────────────────────────────────
    "inherits": "V14_2_CORE_RUNNER_REPLACEMENT",
    "version": "14.3",
    "profile_scope": "HIGH_VOL_LIQUID_OPTIONS_ONLY",
    "status": "LOCAL_OPTIMUM_REPLAY_VALIDATED_NOT_LIVE_PROVEN",
    "default_mode": "PAPER_OR_REPLAY_ONLY",
    "live_trading_enabled": False,

    # ─── Instrument (inherited, unchanged) ───────────────────────────────
    "instrument": "CALL_AND_PUT_DEBIT_SPREADS_ONLY",
    "order_type": "LIMIT_ONLY",
    "order_class": "MLEG",
    "base_signal_mode": "WATCH_ONLY",
    "entry_requires_confirmation": True,
    "entry_requires_spread_quality": True,
    "no_real_money_base_spread": True,

    # ─── Routes (inherited) ──────────────────────────────────────────────
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

    # ─── Watch / Confirmation (OPTIMIZED for high-vol) ───────────────────
    "watch_min_route_score": 0.58,
    "watch_min_ic_spread": 0.025,
    "watch_min_ev_over_debit": 0.10,
    "watch_min_option_liquidity": 0.55,

    "confirm_min_directional_atr": 0.13,   # fast confirm - early entry
    "confirm_max_adverse_atr": 0.21,       # tolerate noise
    "confirm_min_mfe_velocity": 0.35,      # moderate velocity filter

    # ─── Spread Quality (relaxed for wider vol-name spreads) ─────────────
    "max_option_mid_inflation": 0.12,
    "max_composite_spread_pct_of_mid": 0.15,
    "max_limit_chase_pct_of_debit": 0.05,
    "max_move_consumed_pct": 0.50,

    # ─── Package Qualification ───────────────────────────────────────────
    "package_min_q": 0.626,
    "package_min_pwin": 0.581,
    "max_env_stress": 0.30,
    "hi_iv_min_q": 0.700,
    "max_original_debit": 300.00,

    # ─── Package Structure ───────────────────────────────────────────────
    "package_debit_scale": 1.014,
    "max_package_debit": 280.00,
    "core_fraction": 0.720,
    "runner_fraction": 0.280,

    # ─── Package Payoff (OPTIMIZED: wide stop, big target) ───────────────
    "core_target_pct": 0.75,
    "runner_target_pct": 2.50,
    "runner_lock_pct": 0.10,
    "package_stop_pct": -0.21,

    # ─── Fallback Spread (OPTIMIZED) ────────────────────────────────────
    "fallback": "V12_SINGLE_SPREAD",
    "fallback_loss_cut": 0.015,
    "fallback_win_haircut": 0.015,
    "soft_greed_target_pct": 1.00,
    "full_greed_target_pct": 1.50,
    "single_spread_initial_stop_pct": -0.30,

    # ─── Risk / Frequency (OPTIMIZED) ───────────────────────────────────
    "max_trades_per_day": 3,
    "max_trades_per_week": 8,
    "max_same_underlying_trades_per_week": 3,
    "daily_kill_loss_pct": -0.10,
    "weekly_kill_loss_pct": -0.15,
    "max_open_debit_exposure_pct": 0.35,

    # ─── Symbol Gating ───────────────────────────────────────────────────
    "allowed_symbols_initial": ["COIN", "TSLA"],
    "observe_only_symbols": ["MARA", "SPCX"],

    # ─── Promotion Rules ─────────────────────────────────────────────────
    "symbol_promotion_required": True,
    "min_symbol_trades_for_promotion": 20,
    "min_symbol_win_rate": 0.58,
    "min_symbol_profit_factor": 2.0,
    "min_symbol_net_pnl": 0,

    # ─── TSLA Exception (profitable but low win-rate) ────────────────────
    "low_win_high_pf_exception": {
        "enabled": True,
        "min_profit_factor": 2.0,
        "min_positive_pnl": True,
        "max_size_fraction_multiplier": 0.50,
        "requires_route_profitability": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOL POLICY STATE
# ═══════════════════════════════════════════════════════════════════════════

SYMBOL_POLICY_STATE: Dict[str, dict] = {
    "COIN": {
        "mode": "ACTIVE_HIGH_VOL",
        "size_multiplier": 1.00,
        "min_route_pf": 2.0,
        "status": "promoted",
        "reason": "75% WR, 6.10 PF, +$698 backtest",
    },
    "TSLA": {
        "mode": "PROBATION_HIGH_VOL",
        "size_multiplier": 0.50,
        "min_route_pf": 2.0,
        "max_consecutive_losses": 2,
        "status": "conditional",
        "reason": "35.7% WR but 2.19 PF, payoff-ratio carry",
    },
    "MARA": {
        "mode": "OBSERVE_ONLY",
        "size_multiplier": 0.0,
        "status": "disabled",
        "reason": "PF < 1.0, not profitable",
    },
    "SPCX": {
        "mode": "NO_OPTION_SPREAD_MODE",
        "size_multiplier": 0.0,
        "status": "disabled",
        "reason": "No option chain on Alpaca",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# POLICY INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def get_v14_3_config() -> dict:
    """Return a copy of the V14.3 high-vol config."""
    return dict(V14_3_HIGH_VOL_CORE_RUNNER)


def get_symbol_policy(symbol: str) -> dict:
    """Get trading policy for a specific symbol."""
    return SYMBOL_POLICY_STATE.get(symbol.upper(), {
        "mode": "UNKNOWN",
        "size_multiplier": 0.0,
        "status": "not_evaluated",
        "reason": "symbol not in policy state",
    })


def is_symbol_active(symbol: str) -> bool:
    """Check if symbol is allowed for active trading."""
    policy = get_symbol_policy(symbol)
    return policy["mode"] in ("ACTIVE_HIGH_VOL", "PROBATION_HIGH_VOL")


def get_size_multiplier(symbol: str) -> float:
    """Get position size multiplier for a symbol (0.0 to 1.0)."""
    policy = get_symbol_policy(symbol)
    return policy.get("size_multiplier", 0.0)


def can_trade_symbol(symbol: str) -> tuple:
    """
    Full trade permission check.
    Returns (allowed: bool, reason: str, size_mult: float)
    """
    policy = get_symbol_policy(symbol)
    mode = policy.get("mode", "UNKNOWN")

    if mode == "ACTIVE_HIGH_VOL":
        return True, "active", policy.get("size_multiplier", 1.0)
    elif mode == "PROBATION_HIGH_VOL":
        return True, "probation_half_size", policy.get("size_multiplier", 0.5)
    elif mode == "OBSERVE_ONLY":
        return False, "observe_only", 0.0
    elif mode == "NO_OPTION_SPREAD_MODE":
        return False, "no_option_chain", 0.0
    else:
        return False, "unknown_symbol", 0.0


def promote_symbol(symbol: str, win_rate: float, profit_factor: float, net_pnl: float) -> bool:
    """
    Attempt to promote a symbol from observe-only to active/probation.
    Returns True if promoted.
    """
    cfg = V14_3_HIGH_VOL_CORE_RUNNER
    symbol = symbol.upper()

    if win_rate >= cfg["min_symbol_win_rate"] and \
       profit_factor >= cfg["min_symbol_profit_factor"] and \
       net_pnl >= cfg["min_symbol_net_pnl"]:
        SYMBOL_POLICY_STATE[symbol] = {
            "mode": "ACTIVE_HIGH_VOL",
            "size_multiplier": 1.00,
            "min_route_pf": 2.0,
            "status": "promoted",
            "reason": f"Promoted: {win_rate:.0%} WR, {profit_factor:.1f} PF, ${net_pnl:.0f}",
        }
        return True

    # Check low-win high-PF exception (like TSLA)
    exception = cfg.get("low_win_high_pf_exception", {})
    if exception.get("enabled") and \
       profit_factor >= exception.get("min_profit_factor", 2.0) and \
       net_pnl > 0:
        SYMBOL_POLICY_STATE[symbol] = {
            "mode": "PROBATION_HIGH_VOL",
            "size_multiplier": exception.get("max_size_fraction_multiplier", 0.50),
            "min_route_pf": 2.0,
            "max_consecutive_losses": 2,
            "status": "conditional",
            "reason": f"Probation: {win_rate:.0%} WR, {profit_factor:.1f} PF (low-win exception)",
        }
        return True

    return False


def demote_symbol(symbol: str, reason: str):
    """Demote a symbol back to observe-only."""
    symbol = symbol.upper()
    SYMBOL_POLICY_STATE[symbol] = {
        "mode": "OBSERVE_ONLY",
        "size_multiplier": 0.0,
        "status": "demoted",
        "reason": reason,
    }
