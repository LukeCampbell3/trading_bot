"""
V14.2 Backtest on High-Volatility Stocks
=========================================

Runs the V14.2 strategy pipeline on historical 1-min bar data.
Optimizes confirmation and entry parameters for high-vol names like SPCX.

Since SPCX has no listed options, we simulate spread P&L using delta-approximation
from the underlying price action. For MARA/COIN/TSLA we note that real options
would be available in live mode.

Usage: python backtest_v14_2.py
"""

import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as BASE_CFG


@dataclass
class BacktestTrade:
    """Single trade in the backtest."""
    entry_bar: int = 0
    exit_bar: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    side: str = "CALL"  # CALL or PUT
    route: str = ""
    debit: float = 0.0  # simulated spread debit
    pnl_pct: float = 0.0
    pnl_dollar: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    mode: str = "PACKAGE"


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    symbol: str = ""
    config_label: str = ""
    total_bars: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    avg_bars_held: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)


def load_bars(filepath: str) -> pd.DataFrame:
    """Load 1-min bars from CSV."""
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators needed for the strategy."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    n = len(close)

    # VWAP (rolling 100-bar)
    pv = close * volume
    vwap = pd.Series(pv).rolling(100, min_periods=20).sum().values / \
           np.maximum(pd.Series(volume).rolling(100, min_periods=20).sum().values, 1e-10)

    # ATR (14-bar)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(14, min_periods=5).mean().values

    # Trend (20-bar slope via polyfit coefficient)
    trend = np.zeros(n)
    xs = np.arange(20, dtype=float)
    for i in range(19, n):
        y = close[i-19:i+1]
        if len(y) == 20:
            trend[i] = np.polyfit(xs, y, 1)[0]

    # Volume ratio (vs 10-bar mean)
    vol_mean = pd.Series(volume).rolling(10, min_periods=5).mean().values
    vol_ratio = volume / np.maximum(vol_mean, 1.0)

    # EMA 9 and EMA 21 for momentum
    ema9 = pd.Series(close).ewm(span=9).mean().values
    ema21 = pd.Series(close).ewm(span=21).mean().values

    # Rolling high/low of day (approx: 390-bar window)
    hod = pd.Series(high).rolling(390, min_periods=20).max().values
    lod = pd.Series(low).rolling(390, min_periods=20).min().values

    df["vwap"] = vwap
    df["atr"] = atr
    df["trend"] = trend
    df["vol_ratio"] = vol_ratio
    df["ema9"] = ema9
    df["ema21"] = ema21
    df["hod"] = hod
    df["lod"] = lod

    return df


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    cfg: dict,
    config_label: str = "default",
) -> BacktestResult:
    """
    Run V14.2 backtest on historical bars.
    
    Simulates:
    - Watch ticket creation when route qualifies
    - Confirmation after directional move
    - Spread entry (delta-approximated P&L)
    - Core/runner exit management
    """
    result = BacktestResult(symbol=symbol, config_label=config_label)
    result.total_bars = len(df)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vwap = df["vwap"].values
    atr = df["atr"].values
    trend = df["trend"].values
    vol_ratio = df["vol_ratio"].values

    # State
    in_trade = False
    trade: Optional[BacktestTrade] = None
    watch_bar = -1
    watch_price = 0.0
    watch_side = "CALL"
    watch_route = ""
    confirmed = False
    high_since_watch = 0.0
    low_since_watch = 999999.0
    cooldown_until = 0
    trades_today = 0
    day_pnl = 0.0
    current_day = None
    equity_curve = []
    running_equity = 10000.0  # start with $10k

    warmup = 120  # need enough bars for indicators

    for i in range(warmup, len(df)):
        # Daily reset
        ts = df["timestamp"].iloc[i]
        day = ts.date() if hasattr(ts, "date") else str(ts)[:10]
        if day != current_day:
            current_day = day
            trades_today = 0
            day_pnl = 0.0

        # Skip if in cooldown
        if i < cooldown_until:
            continue

        # Skip if daily kill
        if day_pnl / max(running_equity, 1) <= cfg["daily_kill_loss_pct"]:
            continue

        # Skip if max trades hit
        if trades_today >= cfg["max_trades_per_day"]:
            continue

        cur_atr = atr[i] if atr[i] > 0 else 0.01
        cur_price = close[i]
        cur_vwap = vwap[i] if vwap[i] > 0 else cur_price

        # ─── MANAGE OPEN TRADE ───────────────────────────────────────
        if in_trade and trade:
            trade.bars_held += 1

            if trade.side == "CALL":
                move_pct = (cur_price - trade.entry_price) / trade.entry_price
            else:
                move_pct = (trade.entry_price - cur_price) / trade.entry_price

            # Spread P&L approximation: spread gains ~delta * underlying move
            # For ATM debit spread: effective delta ~0.35-0.50
            spread_delta = 0.40
            spread_pnl_pct = move_pct * spread_delta / trade.debit * trade.entry_price

            # Check exits
            exit_reason = ""
            if trade.mode == "PACKAGE":
                # Core target
                if spread_pnl_pct >= cfg["core_target_pct"]:
                    exit_reason = "core_target"
                # Package stop
                elif spread_pnl_pct <= cfg["package_stop_pct"]:
                    exit_reason = "package_stop"
                # Time stop (max 30 bars)
                elif trade.bars_held >= 30:
                    exit_reason = "time_stop"
            else:  # FALLBACK
                if spread_pnl_pct >= cfg["soft_greed_target_pct"]:
                    exit_reason = "fallback_target"
                elif spread_pnl_pct <= cfg["single_spread_initial_stop_pct"]:
                    exit_reason = "fallback_stop"
                elif trade.bars_held >= 20:
                    exit_reason = "time_stop"

            if exit_reason:
                trade.exit_bar = i
                trade.exit_price = cur_price
                trade.pnl_pct = spread_pnl_pct
                trade.pnl_dollar = spread_pnl_pct * trade.debit * 100  # per contract
                trade.exit_reason = exit_reason
                result.trades.append(trade)

                day_pnl += trade.pnl_dollar
                running_equity += trade.pnl_dollar
                equity_curve.append(running_equity)

                in_trade = False
                cooldown_until = i + 3  # 3-bar cooldown
                trade = None
                confirmed = False
                watch_bar = -1
            continue

        # ─── WATCH PHASE ─────────────────────────────────────────────
        if watch_bar < 0:
            # Look for route candidates
            price_vs_vwap = (cur_price - cur_vwap) / cur_atr if cur_atr > 0 else 0

            # CALL: VWAP pullback (price near/below VWAP in uptrend)
            if (-0.6 < price_vs_vwap < 0.3 and trend[i] > 0 and
                vol_ratio[i] > 0.7):
                score = 0.3 + min(0.25, trend[i] * 40) + min(0.15, (vol_ratio[i]-0.7)*0.2)
                if score >= cfg["watch_min_route_score"]:
                    watch_bar = i
                    watch_price = cur_price
                    watch_side = "CALL"
                    watch_route = "VWAP_PULLBACK"
                    high_since_watch = high[i]
                    low_since_watch = low[i]
                    confirmed = False

            # PUT: rejection from VWAP in downtrend
            elif (price_vs_vwap < -0.3 and trend[i] < 0 and vol_ratio[i] > 0.7):
                score = 0.3 + min(0.25, abs(trend[i]) * 40) + min(0.15, (vol_ratio[i]-0.7)*0.2)
                if score >= cfg["watch_min_route_score"]:
                    watch_bar = i
                    watch_price = cur_price
                    watch_side = "PUT"
                    watch_route = "PUT_REJECTION"
                    high_since_watch = high[i]
                    low_since_watch = low[i]
                    confirmed = False

            # CALL: pullback continuation (above VWAP, strong trend)
            elif (price_vs_vwap > 0.2 and trend[i] > 0.002 and vol_ratio[i] > 0.6):
                score = 0.25 + min(0.3, trend[i] * 50) + min(0.15, vol_ratio[i] * 0.1)
                if score >= cfg["watch_min_route_score"]:
                    watch_bar = i
                    watch_price = cur_price
                    watch_side = "CALL"
                    watch_route = "PULLBACK_CONTINUATION"
                    high_since_watch = high[i]
                    low_since_watch = low[i]
                    confirmed = False

        # ─── CONFIRMATION PHASE ──────────────────────────────────────
        elif not confirmed and watch_bar > 0:
            # Update tracking
            high_since_watch = max(high_since_watch, high[i])
            low_since_watch = min(low_since_watch, low[i])

            # Expire if too old
            if i - watch_bar > 20:
                watch_bar = -1
                continue

            # Check confirmation conditions
            if watch_side == "CALL":
                directional_move = cur_price - watch_price
                adverse_move = watch_price - low_since_watch
            else:
                directional_move = watch_price - cur_price
                adverse_move = high_since_watch - watch_price

            dir_atr = directional_move / cur_atr if cur_atr > 0 else 0
            adv_atr = adverse_move / cur_atr if cur_atr > 0 else 0

            # MFE velocity: rate of favorable movement
            bars_since = i - watch_bar
            mfe_vel = dir_atr / max(bars_since, 1) * 5  # normalized

            # VWAP check
            if watch_side == "CALL":
                vwap_ok = cur_price >= cur_vwap * 0.998
            else:
                vwap_ok = cur_price <= cur_vwap * 1.002

            if (dir_atr >= cfg["confirm_min_directional_atr"] and
                adv_atr <= cfg["confirm_max_adverse_atr"] and
                mfe_vel >= cfg["confirm_min_mfe_velocity"] and
                vwap_ok):
                confirmed = True

                # ─── ENTRY ───────────────────────────────────────────
                # Simulate spread debit based on ATR
                debit_per_share = cur_atr * 0.6  # typical spread cost
                debit = debit_per_share  # per share

                # Determine mode
                if (watch_route in cfg["routes_allowed"] and
                    cur_atr / cur_price > 0.002):  # sufficient vol for package
                    mode = "PACKAGE"
                else:
                    mode = "FALLBACK"

                trade = BacktestTrade(
                    entry_bar=i,
                    entry_price=cur_price,
                    side=watch_side,
                    route=watch_route,
                    debit=debit,
                    mode=mode,
                )
                in_trade = True
                trades_today += 1
                watch_bar = -1

    # ─── Compute Results ─────────────────────────────────────────────────
    if result.trades:
        pnls = [t.pnl_pct for t in result.trades]
        dollars = [t.pnl_dollar for t in result.trades]

        result.total_trades = len(result.trades)
        result.wins = sum(1 for p in pnls if p > 0)
        result.losses = sum(1 for p in pnls if p <= 0)
        result.win_rate = result.wins / result.total_trades if result.total_trades > 0 else 0
        result.total_pnl = sum(dollars)
        result.avg_pnl = np.mean(dollars)

        total_wins = sum(d for d in dollars if d > 0)
        total_losses = abs(sum(d for d in dollars if d < 0))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        result.avg_bars_held = np.mean([t.bars_held for t in result.trades])

        # Drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                peak = max(peak, eq)
                dd = (eq - peak) / peak
                max_dd = min(max_dd, dd)
            result.max_drawdown = max_dd

        # Sharpe
        if len(pnls) > 1:
            result.sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

    return result


def main():
    print("=" * 72)
    print("V14.2 BACKTEST - HIGH VOLATILITY STOCK OPTIMIZATION")
    print("=" * 72)

    data_dir = Path("cache/backtest")
    symbols_data = {}

    for f in data_dir.glob("*_1min.csv"):
        sym = f.stem.replace("_1min", "").upper()
        df = load_bars(str(f))
        df = compute_indicators(df)
        symbols_data[sym] = df
        print(f"  Loaded {sym}: {len(df)} bars")

    # ─── Config Variants to Test ─────────────────────────────────────────
    configs = {}

    # Base V14.2 config
    configs["V14.2_BASE"] = dict(BASE_CFG)

    # Optimized for high-vol: lower confirmation thresholds (vol stocks move fast)
    hi_vol = dict(BASE_CFG)
    hi_vol["confirm_min_directional_atr"] = 0.18  # lower from 0.263 (faster confirm)
    hi_vol["confirm_max_adverse_atr"] = 0.20  # wider from 0.151 (allow more noise)
    hi_vol["confirm_min_mfe_velocity"] = 0.35  # lower from 0.487 (don't require fast moves)
    hi_vol["watch_min_route_score"] = 0.62  # lower from 0.706 (more watch opportunities)
    hi_vol["core_target_pct"] = 0.45  # raise from 0.372 (capture more of big moves)
    hi_vol["runner_target_pct"] = 1.5  # raise from 1.155
    hi_vol["package_stop_pct"] = -0.12  # wider from -0.081 (don't get stopped on noise)
    hi_vol["single_spread_initial_stop_pct"] = -0.22  # wider stop
    hi_vol["max_trades_per_day"] = 3  # allow more in volatile markets
    configs["V14.2_HIGH_VOL"] = hi_vol

    # Aggressive high-vol: even more opportunities
    aggressive = dict(hi_vol)
    aggressive["confirm_min_directional_atr"] = 0.12  # very fast confirm
    aggressive["confirm_max_adverse_atr"] = 0.25  # very wide adverse tolerance
    aggressive["watch_min_route_score"] = 0.55  # many more watches
    aggressive["core_target_pct"] = 0.55  # bigger target
    aggressive["package_stop_pct"] = -0.15  # wider stop
    aggressive["max_trades_per_day"] = 4
    configs["V14.2_AGGRESSIVE"] = aggressive

    # Tight high-vol: fewer trades, higher quality
    tight = dict(BASE_CFG)
    tight["confirm_min_directional_atr"] = 0.22
    tight["confirm_max_adverse_atr"] = 0.12
    tight["confirm_min_mfe_velocity"] = 0.45
    tight["watch_min_route_score"] = 0.68
    tight["core_target_pct"] = 0.50
    tight["package_stop_pct"] = -0.09
    tight["single_spread_initial_stop_pct"] = -0.18
    configs["V14.2_TIGHT_HV"] = tight

    # OPTIMIZED: Based on round 1 results - tight entry, wide stop, big target
    optimized = dict(BASE_CFG)
    optimized["confirm_min_directional_atr"] = 0.20  # require real move
    optimized["confirm_max_adverse_atr"] = 0.14  # tight adverse
    optimized["confirm_min_mfe_velocity"] = 0.40  # moderate velocity
    optimized["watch_min_route_score"] = 0.65  # moderate selectivity
    optimized["core_target_pct"] = 0.60  # big target for vol stocks
    optimized["runner_target_pct"] = 1.8  # let runners run on vol
    optimized["package_stop_pct"] = -0.14  # wide stop (vol stocks are noisy)
    optimized["single_spread_initial_stop_pct"] = -0.25  # wide fallback stop
    optimized["soft_greed_target_pct"] = 0.70  # higher greed target
    optimized["max_trades_per_day"] = 3
    optimized["daily_kill_loss_pct"] = -0.08  # wider daily kill for vol
    configs["V14.2_OPTIMIZED_HV"] = optimized

    # SPCX-specific: Very high vol needs very wide stops + strict entries
    spcx_cfg = dict(BASE_CFG)
    spcx_cfg["confirm_min_directional_atr"] = 0.25  # need strong directional move
    spcx_cfg["confirm_max_adverse_atr"] = 0.18  # allow some noise (SPCX is wild)
    spcx_cfg["confirm_min_mfe_velocity"] = 0.35  # don't over-filter on velocity
    spcx_cfg["watch_min_route_score"] = 0.58  # lower bar to get more watches
    spcx_cfg["core_target_pct"] = 0.75  # big target (SPCX moves 10%+ intraday)
    spcx_cfg["runner_target_pct"] = 2.5  # huge runner target
    spcx_cfg["package_stop_pct"] = -0.18  # very wide stop for extreme vol
    spcx_cfg["single_spread_initial_stop_pct"] = -0.30  # ultra wide
    spcx_cfg["soft_greed_target_pct"] = 0.90
    spcx_cfg["max_trades_per_day"] = 3
    spcx_cfg["daily_kill_loss_pct"] = -0.10
    configs["V14.2_SPCX_TUNED"] = spcx_cfg

    # ─── Run Backtests ───────────────────────────────────────────────────
    all_results = []
    print(f"\n{'='*72}")
    print(f"{'Symbol':<8} {'Config':<20} {'Trades':>6} {'WinRate':>8} {'PF':>6} "
          f"{'PnL$':>8} {'Sharpe':>7} {'MaxDD':>7} {'AvgBars':>7}")
    print("-" * 72)

    for sym, df in symbols_data.items():
        for cfg_name, cfg in configs.items():
            result = run_backtest(df, sym, cfg, cfg_name)
            all_results.append(result)

            print(f"{sym:<8} {cfg_name:<20} {result.total_trades:>6} "
                  f"{result.win_rate:>7.1%} {result.profit_factor:>6.2f} "
                  f"${result.total_pnl:>7.0f} {result.sharpe:>7.2f} "
                  f"{result.max_drawdown:>6.1%} {result.avg_bars_held:>7.1f}")

    # ─── Find Best Config Per Symbol ─────────────────────────────────────
    print(f"\n{'='*72}")
    print("BEST CONFIGS BY SYMBOL")
    print(f"{'='*72}")

    for sym in symbols_data.keys():
        sym_results = [r for r in all_results if r.symbol == sym and r.total_trades >= 3]
        if not sym_results:
            print(f"\n  {sym}: insufficient trades across all configs")
            continue
        # Rank by profit factor * win_rate (balanced metric)
        best = max(sym_results, key=lambda r: r.profit_factor * r.win_rate if r.profit_factor < 100 else r.win_rate)
        print(f"\n  {sym} -> {best.config_label}")
        print(f"    Trades: {best.total_trades} | Win rate: {best.win_rate:.1%} | "
              f"PF: {best.profit_factor:.2f} | Sharpe: {best.sharpe:.2f}")
        print(f"    Total PnL: ${best.total_pnl:.0f} | Max DD: {best.max_drawdown:.1%} | "
              f"Avg hold: {best.avg_bars_held:.0f} bars")

        # Route breakdown
        routes = {}
        for t in best.trades:
            routes.setdefault(t.route, []).append(t.pnl_pct)
        print(f"    Routes:")
        for route, pnls in routes.items():
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"      {route}: {len(pnls)} trades, {wr:.0%} win, "
                  f"avg {np.mean(pnls)*100:.1f}%")

    # ─── SPCX-Specific Recommendation ───────────────────────────────────
    print(f"\n{'='*72}")
    print("SPCX OPTIMIZATION SUMMARY")
    print(f"{'='*72}")
    spcx_results = [r for r in all_results if r.symbol == "SPCX" and r.total_trades >= 2]
    if spcx_results:
        best_spcx = max(spcx_results, key=lambda r: r.profit_factor * r.win_rate if r.profit_factor < 100 else r.win_rate)
        print(f"\n  Best config for SPCX: {best_spcx.config_label}")
        print(f"  Win rate: {best_spcx.win_rate:.1%}")
        print(f"  Profit factor: {best_spcx.profit_factor:.2f}")
        print(f"  Total P&L: ${best_spcx.total_pnl:.0f}")
        print(f"  Sharpe: {best_spcx.sharpe:.2f}")
        print(f"  Max drawdown: {best_spcx.max_drawdown:.1%}")
        print(f"  Total trades: {best_spcx.total_trades}")
    else:
        print("\n  SPCX: Not enough trades generated. Data may be too short.")
        print("  Note: SPCX has no listed options - strategy would need equity-mode.")

    # ─── Save Results ────────────────────────────────────────────────────
    results_path = Path("HFT/logs/v14_2/backtest_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "config", "trades", "win_rate", "profit_factor",
                    "total_pnl", "avg_pnl", "sharpe", "max_drawdown", "avg_bars_held"])
        for r in all_results:
            w.writerow([r.symbol, r.config_label, r.total_trades,
                        f"{r.win_rate:.4f}", f"{r.profit_factor:.4f}",
                        f"{r.total_pnl:.2f}", f"{r.avg_pnl:.2f}",
                        f"{r.sharpe:.4f}", f"{r.max_drawdown:.4f}",
                        f"{r.avg_bars_held:.1f}"])
    print(f"\n  Results saved: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
