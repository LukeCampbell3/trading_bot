"""V14.3 high-vol debit-spread paper runner.

This runner talks to Alpaca's real paper Trading API and Market Data API.
It does NOT place live orders. It uses the V14.3 symbol policy (COIN active,
TSLA probation/half-size) and the broker-backed core/runner implementation.

Examples:
    python run_v14_3_options_paper.py --symbol COIN --communication-only
    python run_v14_3_options_paper.py --symbol COIN --poll-interval 60
    python run_v14_3_options_paper.py --symbol TSLA --poll-interval 60 --max-iterations 30
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz

from alpaca_config import AlpacaConfig
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from strategy.v14_3_options_runner import create_v14_3_options_runner
from strategy.v14_3_highvol_config import can_trade_symbol


def _make_clients():
    AlpacaConfig.validate()
    trading = TradingClient(
        AlpacaConfig.API_KEY,
        AlpacaConfig.API_SECRET,
        paper=True,
        url_override=AlpacaConfig.BASE_URL,
    )
    stock = StockHistoricalDataClient(
        AlpacaConfig.API_KEY,
        AlpacaConfig.API_SECRET,
        url_override=AlpacaConfig.DATA_URL,
    )
    options = OptionHistoricalDataClient(
        AlpacaConfig.API_KEY,
        AlpacaConfig.API_SECRET,
    )
    return trading, stock, options


def _bars_frame(stock_client, symbol: str, lookback_minutes: int = 240) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=now - timedelta(minutes=max(lookback_minutes, 180)),
        end=now,
        feed=DataFeed.IEX,
        limit=1000,
    )
    bars = stock_client.get_stock_bars(req)
    df = bars.df.copy()
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol, level=0)
        except Exception:
            df = df.reset_index()
            df = df[df["symbol"] == symbol].set_index("timestamp")
    return df.sort_index()


def _features(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        raise ValueError(f"need >=30 one-minute bars, got {len(df)}")
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    volume = df["volume"].astype(float).to_numpy()

    typical = (high + low + close) / 3.0
    pv = typical * volume
    cum_v = np.cumsum(volume)
    vwap_series = np.cumsum(pv) / np.maximum(cum_v, 1.0)

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = float(np.mean(tr[-14:]))
    xs = np.arange(20, dtype=float)
    trend_slope = float(np.polyfit(xs, close[-20:], 1)[0])
    vol_mean = float(np.mean(volume[-10:]))

    return {
        "price": float(close[-1]),
        "vwap": float(vwap_series[-1]),
        "atr": max(atr, 1e-6),
        "high_of_day": float(np.max(high[-390:])),
        "low_of_day": float(np.min(low[-390:])),
        "trend_slope": trend_slope,
        "volume_ratio": float(volume[-1] / vol_mean) if vol_mean > 0 else 1.0,
        "price_5m_ago": float(close[-6]) if len(close) >= 6 else float(close[-1]),
        "price_15m_ago": float(close[-16]) if len(close) >= 16 else float(close[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V14.3 broker-backed options paper runner")
    parser.add_argument("--symbol", default="COIN", choices=["COIN", "TSLA"])
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--communication-only", action="store_true")
    args = parser.parse_args()

    allowed, reason, size = can_trade_symbol(args.symbol)
    if not allowed:
        print(f"BLOCKED by V14.3 symbol policy: {reason}")
        return 2

    print("=" * 78)
    print("V14.3 HIGH-VOL CORE/RUNNER - ALPACA PAPER OPTIONS")
    print(f"symbol={args.symbol} size_multiplier={size:.2f}")
    print("LIVE TRADING: DISABLED")
    print("=" * 78)

    try:
        trading, stock, options = _make_clients()
        account = trading.get_account()
        print(f"Trading API: connected | status={getattr(account, 'status', '')}")
        print(f"options_approved_level={getattr(account, 'options_approved_level', None)}")
        print(f"options_trading_level={getattr(account, 'options_trading_level', None)}")

        runner = create_v14_3_options_runner(
            args.symbol,
            trading_client=trading,
            option_data_client=options,
            paper_mode=True,
        )

        # Use real underlying price for option communication/preflight.
        df = _bars_frame(stock, args.symbol)
        f = _features(df)
        probe = runner.execution_manager.communication_probe()
        print(f"Broker probe: {probe}")
        spread_probe = runner.chain_fetcher.get_spread_with_quotes(
            args.symbol,
            underlying_price=f["price"],
            side="CALL",
            dte_min=5,
            dte_max=14,
            strike_width=5.0,
        ) if runner.chain_fetcher else None
        if not spread_probe:
            print("Option data probe: FAILED - no tradable quoted spread returned")
            return 3
        long_leg, short_leg, long_contract, short_contract = spread_probe
        print(
            "Option data probe: PASS | "
            f"{long_contract.symbol} {long_leg.bid:.2f}x{long_leg.ask:.2f} / "
            f"{short_contract.symbol} {short_leg.bid:.2f}x{short_leg.ask:.2f}"
        )

        if args.communication_only:
            print("Communication-only mode complete. No order was submitted.")
            return 0

        eastern = pytz.timezone("US/Eastern")
        iteration = 0
        while True:
            iteration += 1
            if args.max_iterations and iteration > args.max_iterations:
                break

            clock = trading.get_clock()
            if not clock.is_open:
                print(f"[{datetime.now(eastern):%H:%M:%S ET}] market closed; no options entries")
                time.sleep(args.poll_interval)
                continue

            df = _bars_frame(stock, args.symbol)
            f = _features(df)
            account = trading.get_account()
            equity = float(account.equity)
            options_bp = getattr(account, "options_buying_power", None)
            buying_power = float(options_bp) if options_bp not in (None, "") else float(account.buying_power)
            runner.update_account(equity=equity, buying_power=buying_power)

            result = runner.evaluate_opportunity(symbol=args.symbol, **f)
            print(
                f"[{datetime.now(eastern):%H:%M:%S ET}] "
                f"price={f['price']:.2f} vwap={f['vwap']:.2f} "
                f"action={result['action']} details={result.get('details', {})}"
            )
            time.sleep(args.poll_interval)

        print(runner.get_status())
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
