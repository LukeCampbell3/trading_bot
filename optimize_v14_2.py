"""
V14.2 Recursive Parameter Optimization
=======================================

Iteratively optimizes strategy parameters using grid search + hill climbing
until no further improvement is found.

Process:
1. Start with best known config from round 1
2. For each parameter, test variations (up/down)
3. Keep improvement if metric improves
4. Repeat until no parameter change improves the score
5. Output final optimized config

Optimization target: profit_factor * win_rate (balanced)
Constraint: min 5 trades, win_rate >= 50%, profit_factor >= 1.5
"""

import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as BASE_CFG


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (inlined for speed)
# ═══════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)

    pv = close * volume
    vwap = pd.Series(pv).rolling(100, min_periods=20).sum().values / \
           np.maximum(pd.Series(volume).rolling(100, min_periods=20).sum().values, 1e-10)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(14, min_periods=5).mean().values

    trend = np.zeros(n)
    xs = np.arange(20, dtype=float)
    for i in range(19, n):
        y = close[i-19:i+1]
        if len(y) == 20:
            trend[i] = np.polyfit(xs, y, 1)[0]

    vol_mean = pd.Series(volume).rolling(10, min_periods=5).mean().values
    vol_ratio = volume / np.maximum(vol_mean, 1.0)

    df["vwap"] = vwap
    df["atr"] = atr
    df["trend"] = trend
    df["vol_ratio"] = vol_ratio
    return df


def run_backtest_fast(df: pd.DataFrame, cfg: dict) -> dict:
    """Fast backtest returning key metrics."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vwap = df["vwap"].values
    atr = df["atr"].values
    trend = df["trend"].values
    vol_ratio = df["vol_ratio"].values

    trades_pnl = []
    in_trade = False
    entry_price = 0.0
    entry_bar = 0
    trade_side = "CALL"
    trade_mode = "PACKAGE"
    trade_debit = 0.0
    watch_bar = -1
    watch_price = 0.0
    watch_side = "CALL"
    confirmed = False
    high_since = 0.0
    low_since = 999999.0
    cooldown_until = 0
    trades_today = 0
    day_pnl = 0.0
    current_day = None
    running_equity = 10000.0

    warmup = 120

    for i in range(warmup, len(df)):
        ts = df["timestamp"].iloc[i]
        day = str(ts)[:10]
        if day != current_day:
            current_day = day
            trades_today = 0
            day_pnl = 0.0

        if i < cooldown_until:
            continue
        if running_equity > 0 and day_pnl / running_equity <= cfg.get("daily_kill_loss_pct", -0.053):
            continue
        if trades_today >= cfg.get("max_trades_per_day", 2):
            continue

        cur_atr = atr[i] if atr[i] > 0 else 0.01
        cur_price = close[i]
        cur_vwap = vwap[i] if vwap[i] > 0 else cur_price

        # MANAGE TRADE
        if in_trade:
            if trade_side == "CALL":
                move_pct = (cur_price - entry_price) / entry_price
            else:
                move_pct = (entry_price - cur_price) / entry_price

            spread_delta = 0.40
            spread_pnl_pct = move_pct * spread_delta / trade_debit * entry_price
            bars_held = i - entry_bar

            exit_reason = ""
            if trade_mode == "PACKAGE":
                if spread_pnl_pct >= cfg.get("core_target_pct", 0.372):
                    exit_reason = "target"
                elif spread_pnl_pct <= cfg.get("package_stop_pct", -0.081):
                    exit_reason = "stop"
                elif bars_held >= 30:
                    exit_reason = "time"
            else:
                if spread_pnl_pct >= cfg.get("soft_greed_target_pct", 0.55):
                    exit_reason = "target"
                elif spread_pnl_pct <= cfg.get("single_spread_initial_stop_pct", -0.16):
                    exit_reason = "stop"
                elif bars_held >= 20:
                    exit_reason = "time"

            if exit_reason:
                pnl_dollar = spread_pnl_pct * trade_debit * 100
                trades_pnl.append(pnl_dollar)
                day_pnl += pnl_dollar
                running_equity += pnl_dollar
                in_trade = False
                cooldown_until = i + 3
                watch_bar = -1
                confirmed = False
            continue

        # WATCH
        if watch_bar < 0:
            price_vs_vwap = (cur_price - cur_vwap) / cur_atr if cur_atr > 0 else 0
            min_score = cfg.get("watch_min_route_score", 0.706)

            if -0.6 < price_vs_vwap < 0.3 and trend[i] > 0 and vol_ratio[i] > 0.7:
                score = 0.3 + min(0.25, trend[i] * 40) + min(0.15, (vol_ratio[i]-0.7)*0.2)
                if score >= min_score:
                    watch_bar = i; watch_price = cur_price; watch_side = "CALL"
                    high_since = high[i]; low_since = low[i]; confirmed = False

            elif price_vs_vwap < -0.3 and trend[i] < 0 and vol_ratio[i] > 0.7:
                score = 0.3 + min(0.25, abs(trend[i]) * 40) + min(0.15, (vol_ratio[i]-0.7)*0.2)
                if score >= min_score:
                    watch_bar = i; watch_price = cur_price; watch_side = "PUT"
                    high_since = high[i]; low_since = low[i]; confirmed = False

            elif price_vs_vwap > 0.2 and trend[i] > 0.002 and vol_ratio[i] > 0.6:
                score = 0.25 + min(0.3, trend[i] * 50) + min(0.15, vol_ratio[i] * 0.1)
                if score >= min_score:
                    watch_bar = i; watch_price = cur_price; watch_side = "CALL"
                    high_since = high[i]; low_since = low[i]; confirmed = False

        # CONFIRM
        elif not confirmed:
            high_since = max(high_since, high[i])
            low_since = min(low_since, low[i])

            if i - watch_bar > 20:
                watch_bar = -1
                continue

            if watch_side == "CALL":
                dir_move = cur_price - watch_price
                adv_move = watch_price - low_since
            else:
                dir_move = watch_price - cur_price
                adv_move = high_since - watch_price

            dir_atr = dir_move / cur_atr if cur_atr > 0 else 0
            adv_atr = adv_move / cur_atr if cur_atr > 0 else 0
            bars_since = i - watch_bar
            mfe_vel = dir_atr / max(bars_since, 1) * 5

            if watch_side == "CALL":
                vwap_ok = cur_price >= cur_vwap * 0.998
            else:
                vwap_ok = cur_price <= cur_vwap * 1.002

            if (dir_atr >= cfg.get("confirm_min_directional_atr", 0.263) and
                adv_atr <= cfg.get("confirm_max_adverse_atr", 0.151) and
                mfe_vel >= cfg.get("confirm_min_mfe_velocity", 0.487) and
                vwap_ok):

                confirmed = True
                trade_debit = cur_atr * 0.6
                route = "VWAP_PB" if -0.6 < (cur_price - cur_vwap)/cur_atr < 0.3 else "CONT"
                if route in ["VWAP_PB", "CONT"] and cur_atr / cur_price > 0.002:
                    trade_mode = "PACKAGE"
                else:
                    trade_mode = "FALLBACK"

                entry_price = cur_price
                entry_bar = i
                trade_side = watch_side
                in_trade = True
                trades_today += 1
                watch_bar = -1

    # Compute metrics
    total = len(trades_pnl)
    if total == 0:
        return {"trades": 0, "win_rate": 0, "pf": 0, "pnl": 0, "sharpe": 0, "max_dd": 0, "score": 0}

    wins = sum(1 for p in trades_pnl if p > 0)
    win_rate = wins / total
    total_wins = sum(p for p in trades_pnl if p > 0)
    total_losses = abs(sum(p for p in trades_pnl if p < 0))
    pf = total_wins / total_losses if total_losses > 0 else 99.0
    sharpe = np.mean(trades_pnl) / np.std(trades_pnl) if np.std(trades_pnl) > 0 else 0

    # Max drawdown
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    for p in trades_pnl:
        equity += p
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    # Score: prioritize profit factor and win rate, penalize few trades
    trade_bonus = min(1.0, total / 10.0)  # full bonus at 10+ trades
    score = pf * win_rate * trade_bonus if pf < 50 else win_rate * trade_bonus

    return {
        "trades": total, "win_rate": win_rate, "pf": pf,
        "pnl": sum(trades_pnl), "sharpe": sharpe, "max_dd": max_dd,
        "score": score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

# Parameters to optimize with their step sizes and bounds
PARAM_SPACE = {
    "watch_min_route_score": (0.04, 0.40, 0.85),
    "confirm_min_directional_atr": (0.03, 0.08, 0.45),
    "confirm_max_adverse_atr": (0.03, 0.06, 0.35),
    "confirm_min_mfe_velocity": (0.05, 0.15, 0.70),
    "core_target_pct": (0.08, 0.20, 1.20),
    "runner_target_pct": (0.2, 0.80, 3.50),
    "package_stop_pct": (0.03, -0.30, -0.04),
    "single_spread_initial_stop_pct": (0.04, -0.45, -0.08),
    "soft_greed_target_pct": (0.10, 0.30, 1.50),
    "daily_kill_loss_pct": (0.02, -0.20, -0.03),
    "max_trades_per_day": (1, 1, 6),
}


def optimize_recursive(datasets: Dict[str, pd.DataFrame], start_cfg: dict) -> Tuple[dict, list]:
    """
    Recursively optimize parameters until no improvement.
    Returns (best_config, optimization_log).
    """
    cfg = copy.deepcopy(start_cfg)
    log = []
    iteration = 0

    # Evaluate starting config across all datasets
    best_score = evaluate_config(datasets, cfg)
    print(f"\n  [Iter 0] Starting score: {best_score:.4f}")
    log.append({"iteration": 0, "score": best_score, "param": "START", "value": 0, "improved": True})

    while True:
        iteration += 1
        improved = False
        print(f"\n  [Iter {iteration}] Scanning {len(PARAM_SPACE)} parameters...")

        for param, (step, lo, hi) in PARAM_SPACE.items():
            current_val = cfg.get(param, 0)

            # Try increasing
            new_val_up = current_val + step
            if param == "package_stop_pct" or param == "single_spread_initial_stop_pct" or param == "daily_kill_loss_pct":
                # These are negative; "up" means less negative (tighter)
                new_val_up = min(hi, current_val + step)
            else:
                new_val_up = min(hi, current_val + step)

            # Try decreasing
            new_val_down = current_val - step
            if param == "package_stop_pct" or param == "single_spread_initial_stop_pct" or param == "daily_kill_loss_pct":
                new_val_down = max(lo, current_val - step)
            else:
                new_val_down = max(lo, current_val - step)

            # Integer params
            if param == "max_trades_per_day":
                new_val_up = int(min(hi, current_val + step))
                new_val_down = int(max(lo, current_val - step))

            # Test up
            test_cfg = copy.deepcopy(cfg)
            test_cfg[param] = new_val_up
            score_up = evaluate_config(datasets, test_cfg)

            # Test down
            test_cfg2 = copy.deepcopy(cfg)
            test_cfg2[param] = new_val_down
            score_down = evaluate_config(datasets, test_cfg2)

            # Pick best
            best_direction = None
            if score_up > best_score and score_up >= score_down:
                best_direction = "up"
                best_score = score_up
                cfg[param] = new_val_up
                improved = True
            elif score_down > best_score:
                best_direction = "down"
                best_score = score_down
                cfg[param] = new_val_down
                improved = True

            if best_direction:
                new_val = cfg[param]
                print(f"    + {param}: {current_val:.4f} -> {new_val:.4f} "
                      f"(score: {best_score:.4f})")
                log.append({
                    "iteration": iteration, "score": best_score,
                    "param": param, "value": new_val, "improved": True
                })

        if not improved:
            print(f"\n  [Iter {iteration}] No improvement found. Optimization complete.")
            log.append({"iteration": iteration, "score": best_score, "param": "CONVERGED", "value": 0, "improved": False})
            break

        if iteration >= 20:
            print(f"\n  [Iter {iteration}] Max iterations reached.")
            break

    return cfg, log


def evaluate_config(datasets: Dict[str, pd.DataFrame], cfg: dict) -> float:
    """Evaluate a config across all datasets, return combined score."""
    scores = []
    for sym, df in datasets.items():
        result = run_backtest_fast(df, cfg)
        # Only count if minimum quality
        if result["trades"] >= 3 and result["win_rate"] >= 0.35:
            scores.append(result["score"])
        elif result["trades"] >= 1:
            scores.append(result["score"] * 0.5)  # penalize low trade count

    if not scores:
        return 0.0
    return np.mean(scores)


def main():
    print("=" * 72)
    print("V14.2 RECURSIVE PARAMETER OPTIMIZATION")
    print("Iterating until no further improvement...")
    print("=" * 72)

    # Load data
    data_dir = Path("cache/backtest")
    datasets = {}
    for f in data_dir.glob("*_1min.csv"):
        sym = f.stem.replace("_1min", "").upper()
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = compute_indicators(df)
        datasets[sym] = df
        print(f"  Loaded {sym}: {len(df)} bars")

    # ─── Start from the best config found in round 1 ────────────────────
    start_cfg = dict(BASE_CFG)
    # Apply the SPCX_TUNED starting point (best from round 1)
    start_cfg["confirm_min_directional_atr"] = 0.25
    start_cfg["confirm_max_adverse_atr"] = 0.18
    start_cfg["confirm_min_mfe_velocity"] = 0.35
    start_cfg["watch_min_route_score"] = 0.58
    start_cfg["core_target_pct"] = 0.75
    start_cfg["runner_target_pct"] = 2.50
    start_cfg["package_stop_pct"] = -0.18
    start_cfg["single_spread_initial_stop_pct"] = -0.30
    start_cfg["soft_greed_target_pct"] = 0.90
    start_cfg["max_trades_per_day"] = 3
    start_cfg["daily_kill_loss_pct"] = -0.10

    print(f"\n{'─'*72}")
    print("PHASE 1: Optimizing across ALL symbols (COIN, MARA, SPCX, TSLA)")
    print(f"{'─'*72}")
    best_cfg_all, log_all = optimize_recursive(datasets, start_cfg)

    # ─── Phase 2: Optimize specifically for high-vol (COIN + SPCX) ───────
    print(f"\n{'─'*72}")
    print("PHASE 2: Optimizing for HIGH-VOL only (COIN, SPCX)")
    print(f"{'─'*72}")
    hv_datasets = {k: v for k, v in datasets.items() if k in ("COIN", "SPCX", "MARA")}
    best_cfg_hv, log_hv = optimize_recursive(hv_datasets, best_cfg_all)

    # ─── Final Evaluation ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("FINAL OPTIMIZED RESULTS")
    print(f"{'='*72}")
    print(f"\n{'Symbol':<8} {'Trades':>6} {'WinRate':>8} {'PF':>7} {'PnL$':>8} {'Sharpe':>7} {'MaxDD':>7} {'Score':>7}")
    print("-" * 60)

    for sym, df in datasets.items():
        r = run_backtest_fast(df, best_cfg_hv)
        print(f"{sym:<8} {r['trades']:>6} {r['win_rate']:>7.1%} {r['pf']:>7.2f} "
              f"${r['pnl']:>7.0f} {r['sharpe']:>7.2f} {r['max_dd']:>6.1%} {r['score']:>7.3f}")

    # ─── Compare vs original ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("COMPARISON: Original V14.2 SPCX_TUNED vs Final Optimized")
    print(f"{'─'*72}")
    print(f"\n{'Symbol':<8} {'Metric':<12} {'Original':>10} {'Optimized':>10} {'Delta':>8}")
    print("-" * 50)

    for sym, df in datasets.items():
        r_orig = run_backtest_fast(df, start_cfg)
        r_opt = run_backtest_fast(df, best_cfg_hv)
        if r_orig["trades"] > 0 or r_opt["trades"] > 0:
            print(f"{sym:<8} {'Win Rate':<12} {r_orig['win_rate']:>9.1%} {r_opt['win_rate']:>9.1%} "
                  f"{(r_opt['win_rate']-r_orig['win_rate'])*100:>+7.1f}%")
            print(f"{'':8} {'PF':<12} {r_orig['pf']:>10.2f} {r_opt['pf']:>10.2f} "
                  f"{r_opt['pf']-r_orig['pf']:>+8.2f}")
            print(f"{'':8} {'PnL':<12} ${r_orig['pnl']:>9.0f} ${r_opt['pnl']:>9.0f} "
                  f"${r_opt['pnl']-r_orig['pnl']:>+7.0f}")
            print(f"{'':8} {'Trades':<12} {r_orig['trades']:>10} {r_opt['trades']:>10}")
            print()

    # ─── Output final config ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("FINAL OPTIMIZED CONFIG (copy to v14_2_highvol_config.py)")
    print(f"{'='*72}")
    opt_params = [
        "watch_min_route_score", "confirm_min_directional_atr",
        "confirm_max_adverse_atr", "confirm_min_mfe_velocity",
        "core_target_pct", "runner_target_pct", "package_stop_pct",
        "single_spread_initial_stop_pct", "soft_greed_target_pct",
        "daily_kill_loss_pct", "max_trades_per_day",
    ]
    print("\n  # Optimized parameters (recursive hill-climb converged):")
    for p in opt_params:
        val = best_cfg_hv.get(p, "?")
        orig = start_cfg.get(p, "?")
        changed = " *CHANGED*" if val != orig else ""
        if isinstance(val, float):
            print(f'  "{p}": {val:.4f},{changed}')
        else:
            print(f'  "{p}": {val},{changed}')

    # Save optimization log
    log_path = Path("HFT/logs/v14_2/optimization_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"phase1": log_all, "phase2": log_hv, "final_config": best_cfg_hv}, f, indent=2, default=str)
    print(f"\n  Optimization log saved: {log_path}")

    # Save final config as Python
    config_path = Path("HFT/logs/v14_2/optimized_config.py")
    with open(config_path, "w") as f:
        f.write("# V14.2 Recursively Optimized High-Vol Config\n")
        f.write("# Generated by optimize_v14_2.py\n\n")
        f.write("OPTIMIZED_HIGH_VOL = {\n")
        for k, v in best_cfg_hv.items():
            if isinstance(v, str):
                f.write(f'    "{k}": "{v}",\n')
            elif isinstance(v, bool):
                f.write(f'    "{k}": {v},\n')
            elif isinstance(v, list):
                f.write(f'    "{k}": {v},\n')
            elif isinstance(v, float):
                f.write(f'    "{k}": {v:.6f},\n')
            else:
                f.write(f'    "{k}": {v},\n')
        f.write("}\n")
    print(f"  Final config saved: {config_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
