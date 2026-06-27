"""
V14.2 Core Runner - Paper/Simulation Mode Entry Point
=====================================================
Status: V14_2_CORE_RUNNER_REPLACEMENT_SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN

Usage:
    python run_v14_2_paper.py [--symbol AAPL] [--poll-interval 60]

This runs the V14.2 strategy in paper mode, integrated with the existing
Alpaca trading client for position tracking, cash accounting, and data feeds.

LIVE TRADING IS DISABLED. This is a paper/simulation runner only.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz

from alpaca_config import AlpacaConfig
from alpaca_trader import AlpacaTrader
from strategy.v14_2_core_runner import V14_2_CoreRunner
from strategy.v14_2_config import get_config


def main():
    parser = argparse.ArgumentParser(description="V14.2 Core Runner - Paper Mode")
    parser.add_argument("--symbol", default="AAPL", help="Symbol to trade (default: AAPL)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--max-iterations", type=int, default=0, help="Max iterations (0=infinite)")
    args = parser.parse_args()

    print("=" * 72)
    print("V14.2 CORE RUNNER - PAPER/SIMULATION MODE")
    print("Status: SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN")
    print(f"Symbol: {args.symbol}")
    print(f"Poll interval: {args.poll_interval}s")
    print("LIVE TRADING: DISABLED")
    print("=" * 72)

    # ─── Initialize existing Alpaca trader for data/account access ────────
    try:
        trader = AlpacaTrader(
            symbol=args.symbol,
            model_path="HFT/model/instinct_model.keras",
            scaler_path="HFT/model/scaler.pkl",
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize AlpacaTrader: {e}")
        print("Continuing with V14.2 in standalone paper mode (no live data feed).")
        trader = None

    # ─── Initialize V14.2 Core Runner ────────────────────────────────────
    config = get_config()
    trading_client = trader.trading_client if trader else None

    # Set up option data client for real quotes
    option_data_client = None
    if trading_client:
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            option_data_client = OptionHistoricalDataClient(
                AlpacaConfig.API_KEY,
                AlpacaConfig.API_SECRET,
            )
            print("[V14.2] Option data client: CONNECTED")
        except Exception as e:
            print(f"[V14.2] Option data client failed: {e}")

    runner = V14_2_CoreRunner(
        trading_client=trading_client,
        option_data_client=option_data_client,
        config=config,
        paper_mode=True,
        log_dir="HFT/logs/v14_2",
    )

    # ─── Main Loop ───────────────────────────────────────────────────────
    eastern = pytz.timezone("US/Eastern")
    iteration = 0

    print(f"\n[V14.2] Starting paper trading loop...")
    print(f"[V14.2] Logs: HFT/logs/v14_2/")
    print(f"[V14.2] Press Ctrl+C to stop.\n")

    try:
        while True:
            iteration += 1
            if args.max_iterations > 0 and iteration > args.max_iterations:
                print(f"[V14.2] Max iterations ({args.max_iterations}) reached. Stopping.")
                break

            now_et = datetime.now(eastern)
            print(f"[V14.2] Iteration {iteration} | {now_et.strftime('%H:%M:%S ET')}")

            # ─── Get account info ────────────────────────────────────────
            try:
                if trader:
                    account = trader.get_account_info()
                    runner.update_account(
                        equity=account["equity"],
                        buying_power=account["buying_power"],
                    )
                else:
                    runner.update_account(equity=100000.0, buying_power=50000.0)
            except Exception as e:
                print(f"  [WARN] Account fetch failed: {e}")
                runner.update_account(equity=100000.0, buying_power=50000.0)

            # ─── New day check ───────────────────────────────────────────
            runner.on_new_day()

            # ─── Get market data ─────────────────────────────────────────
            try:
                if trader:
                    trader._poll_latest_rest()
                    buf_data = trader.buf_sym.to_arrays()
                    if not buf_data or len(buf_data.get("close", [])) < 20:
                        print("  [SKIP] Insufficient data in buffer")
                        time.sleep(args.poll_interval)
                        continue

                    close = buf_data["close"]
                    vwap = buf_data["vwap"]
                    high = buf_data["high"]
                    low = buf_data["low"]
                    volume = buf_data["volume"]

                    price = float(close[-1])
                    current_vwap = float(vwap[-1])

                    # Compute ATR (14-bar)
                    import numpy as np
                    tr_arr = np.maximum(
                        high[-15:] - low[-15:],
                        np.maximum(
                            np.abs(high[-15:] - np.roll(close[-15:], 1)),
                            np.abs(low[-15:] - np.roll(close[-15:], 1)),
                        ),
                    )
                    atr = float(np.mean(tr_arr[1:]))  # skip first (roll artifact)

                    # Compute trend slope
                    if len(close) >= 20:
                        xs = np.arange(20)
                        trend_slope = float(np.polyfit(xs, close[-20:], 1)[0])
                    else:
                        trend_slope = 0.0

                    # Volume ratio
                    vol_mean = float(np.mean(volume[-10:])) if len(volume) >= 10 else 1.0
                    vol_ratio = float(volume[-1]) / vol_mean if vol_mean > 0 else 1.0

                    # Prices at various lookbacks
                    price_5m_ago = float(close[-6]) if len(close) >= 6 else price
                    price_15m_ago = float(close[-16]) if len(close) >= 16 else price

                    # Day range
                    high_of_day = float(np.max(high[-390:])) if len(high) >= 390 else float(np.max(high))
                    low_of_day = float(np.min(low[-390:])) if len(low) >= 390 else float(np.min(low))

                else:
                    # Standalone mode with synthetic data
                    price = 150.0
                    current_vwap = 149.5
                    atr = 1.5
                    high_of_day = 151.0
                    low_of_day = 148.5
                    trend_slope = 0.003
                    vol_ratio = 1.1
                    price_5m_ago = 149.8
                    price_15m_ago = 149.0

            except Exception as e:
                print(f"  [WARN] Data fetch failed: {e}")
                time.sleep(args.poll_interval)
                continue

            # ─── Evaluate opportunity ────────────────────────────────────
            result = runner.evaluate_opportunity(
                symbol=args.symbol,
                price=price,
                vwap=current_vwap,
                atr=atr,
                high_of_day=high_of_day,
                low_of_day=low_of_day,
                trend_slope=trend_slope,
                volume_ratio=vol_ratio,
                price_5m_ago=price_5m_ago,
                price_15m_ago=price_15m_ago,
            )

            action = result.get("action", "NONE")
            if action != "NONE":
                print(f"  [ACTION] {action} | {result.get('details', {})}")
            else:
                print(f"  [SCAN] No opportunity | price={price:.2f} vwap={current_vwap:.2f}")

            # ─── Status summary ──────────────────────────────────────────
            status = runner.get_status()
            watches = status["active_watches"]
            if watches > 0:
                print(f"  [STATUS] Active watches: {watches}")

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(f"\n[V14.2] Stopped by user after {iteration} iterations.")

    # ─── Final Status ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL STATUS")
    print("=" * 72)
    status = runner.get_status()
    for k, v in status.items():
        if k != "route_stats":
            print(f"  {k}: {v}")
    print(f"\n  Label: {runner.LABEL}")
    print("=" * 72)


if __name__ == "__main__":
    main()
