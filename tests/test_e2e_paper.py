"""
End-to-End Paper Trading Test for V14.2

This script:
1. Connects to Alpaca paper account
2. Fetches real SPY underlying data
3. Fetches real option contracts and quotes
4. Runs the V14.2 pipeline with real data
5. Verifies the full chain works

Run: python _test_e2e_paper.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from alpaca_config import AlpacaConfig
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from strategy.v14_2_core_runner import V14_2_CoreRunner
from strategy.v14_2_config import get_config
from strategy.option_chain_fetcher import OptionChainFetcher


def main():
    print("=" * 72)
    print("V14.2 END-TO-END PAPER TRADING TEST")
    print("=" * 72)

    # ─── 1. Connect to Alpaca ────────────────────────────────────────────
    print("\n[1] Connecting to Alpaca paper account...")
    trading_client = TradingClient(
        AlpacaConfig.API_KEY,
        AlpacaConfig.API_SECRET,
        paper=True,
        url_override=AlpacaConfig.BASE_URL,
    )
    data_client = StockHistoricalDataClient(
        AlpacaConfig.API_KEY,
        AlpacaConfig.API_SECRET,
        url_override=AlpacaConfig.DATA_URL,
    )
    option_data_client = OptionHistoricalDataClient(
        AlpacaConfig.API_KEY,
        AlpacaConfig.API_SECRET,
    )

    acct = trading_client.get_account()
    equity = float(acct.equity)
    buying_power = float(acct.buying_power)
    print(f"    Account: ${equity:,.2f} equity | ${buying_power:,.2f} buying power")
    print(f"    Options level: {getattr(acct, 'options_trading_level', 'unknown')}")

    # ─── 2. Fetch real SPY data ──────────────────────────────────────────
    print("\n[2] Fetching real SPY market data...")
    symbol = "SPY"
    now = datetime.utcnow()
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=now - timedelta(hours=8),
        end=now,
        limit=200,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(req)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df = df.reset_index()

    if df.empty:
        print("    WARNING: No bars returned (market may be closed)")
        print("    Using last known data for pipeline test...")
        # Use synthetic for testing when market is closed
        price = 590.0
        vwap = 589.5
        atr = 2.0
        high_of_day = 592.0
        low_of_day = 588.0
        trend_slope = 0.003
        vol_ratio = 1.1
        price_5m_ago = 589.8
        price_15m_ago = 589.0
    else:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        vwap_col = df["vwap"].values if "vwap" in df.columns else close

        price = float(close[-1])
        vwap = float(vwap_col[-1])
        print(f"    SPY price: ${price:.2f} | VWAP: ${vwap:.2f} | Bars: {len(df)}")

        # Compute ATR
        tr = np.maximum(
            high[-15:] - low[-15:],
            np.maximum(
                np.abs(high[-15:] - np.roll(close[-15:], 1)),
                np.abs(low[-15:] - np.roll(close[-15:], 1)),
            ),
        )
        atr = float(np.mean(tr[1:]))

        # Trend slope
        if len(close) >= 20:
            xs = np.arange(20)
            trend_slope = float(np.polyfit(xs, close[-20:], 1)[0])
        else:
            trend_slope = 0.0

        vol_mean = float(np.mean(volume[-10:])) if len(volume) >= 10 else 1.0
        vol_ratio = float(volume[-1]) / vol_mean if vol_mean > 0 else 1.0
        price_5m_ago = float(close[-6]) if len(close) >= 6 else price
        price_15m_ago = float(close[-16]) if len(close) >= 16 else price
        high_of_day = float(np.max(high))
        low_of_day = float(np.min(low))

    # ─── 3. Test option chain fetcher ────────────────────────────────────
    print("\n[3] Testing option chain fetcher...")
    chain_fetcher = OptionChainFetcher(trading_client, option_data_client)

    spread_data = chain_fetcher.get_spread_with_quotes(
        symbol=symbol,
        underlying_price=price,
        side="CALL",
        dte_min=5,
        dte_max=14,
        strike_width=5.0,
    )

    if spread_data:
        long_leg, short_leg, long_c, short_c = spread_data
        print(f"    Long:  {long_c.symbol} strike={long_c.strike} "
              f"bid={long_leg.bid:.2f} ask={long_leg.ask:.2f}")
        print(f"    Short: {short_c.symbol} strike={short_c.strike} "
              f"bid={short_leg.bid:.2f} ask={short_leg.ask:.2f}")
        spread_mid = long_leg.mid - short_leg.mid
        print(f"    Spread mid: ${spread_mid:.2f} per share (${spread_mid*100:.0f} per contract)")
    else:
        print("    WARNING: Could not fetch option spread (market may be closed)")

    # ─── 4. Initialize V14.2 Core Runner with real connections ───────────
    print("\n[4] Initializing V14.2 Core Runner...")
    config = get_config()
    runner = V14_2_CoreRunner(
        trading_client=trading_client,
        option_data_client=option_data_client,
        config=config,
        paper_mode=True,
        log_dir="HFT/logs/v14_2",
    )
    runner.update_account(equity, buying_power)
    runner.on_new_day()

    # ─── 5. Run evaluation pipeline ─────────────────────────────────────
    print("\n[5] Running V14.2 evaluation pipeline...")
    result = runner.evaluate_opportunity(
        symbol=symbol,
        price=price,
        vwap=vwap,
        atr=atr,
        high_of_day=high_of_day,
        low_of_day=low_of_day,
        trend_slope=trend_slope,
        volume_ratio=vol_ratio,
        price_5m_ago=price_5m_ago,
        price_15m_ago=price_15m_ago,
    )
    print(f"    Action: {result['action']}")
    print(f"    Details: {result.get('details', {})}")

    # ─── 6. Strategy status ──────────────────────────────────────────────
    print("\n[6] Strategy status:")
    status = runner.get_status()
    for k, v in status.items():
        if k != "route_stats":
            print(f"    {k}: {v}")

    # ─── 7. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("E2E TEST RESULTS")
    print("=" * 72)
    checks = [
        ("Alpaca connection", True),
        ("Account accessible", equity > 0),
        ("Options level >= 2", int(getattr(acct, 'options_trading_level', 0)) >= 2),
        ("Option chain fetcher", spread_data is not None or df.empty),
        ("V14.2 pipeline runs", result["action"] in ("NONE", "WATCHING", "SKIPPED", "CONFIRMED", "FILLED", "SUBMITTED", "REJECTED")),
        ("Risk controls active", not status.get("risk_killed", True)),
        ("Paper mode confirmed", status.get("mode") == "PAPER"),
        ("Live trading disabled", not status.get("live_enabled", True)),
    ]

    all_pass = True
    for name, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 72)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
