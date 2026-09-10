"""V14.4 warm-started paper entrypoint for the V14.3 options strategy.

The strategy remains V14.3; V14.4 is an infrastructure upgrade. Prior bars seed
ATR/trend/volume immediately, while VWAP/HOD/LOD and 5m/15m continuation are
strictly current-session features. The only remaining opening delay is the
minimum session-structure requirement (default 5 bars), not a 20-30 minute
indicator warm-up.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz

from alpaca_config import AlpacaConfig
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Sort

from strategy.v14_3_options_runner import create_v14_3_options_runner
from strategy.v14_3_highvol_config import can_trade_symbol
from strategy.warm_start_v14_4 import SessionWarmFeatureEngine


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


def _warm_bars_frame(stock_client, symbol: str, days: int = 5) -> pd.DataFrame:
    """Fetch newest-first so the API limit always preserves today's bars."""
    now = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=now - timedelta(days=max(3, days)),
        end=now,
        feed=DataFeed.IEX,
        sort=Sort.DESC,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="V14.4 zero-wait V14.3 Alpaca paper options")
    parser.add_argument("--symbol", default="COIN", choices=["COIN", "TSLA"])
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--communication-only", action="store_true")
    parser.add_argument("--min-session-bars", type=int, default=5,
                        help="True intraday structure bars required before entries (default 5)")
    args = parser.parse_args()

    allowed, reason, size = can_trade_symbol(args.symbol)
    if not allowed:
        print(f"BLOCKED by V14.3 symbol policy: {reason}")
        return 2

    print("=" * 82)
    print("V14.4 ZERO-WAIT WARM START + V14.3 HIGH-VOL CORE/RUNNER")
    print(f"symbol={args.symbol} size_multiplier={size:.2f} min_session_bars={args.min_session_bars}")
    print("LIVE TRADING: DISABLED")
    print("=" * 82)

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
        probe = runner.execution_manager.communication_probe()
        print(f"Broker probe: {probe}")

        if args.communication_only:
            # Option-contract/quote probe can run without feature warm-up.
            df = _warm_bars_frame(stock, args.symbol)
            if df.empty:
                print("Stock data probe: FAILED")
                return 3
            px = float(df["close"].iloc[-1])
            spread = runner.chain_fetcher.get_spread_with_quotes(
                args.symbol, underlying_price=px, side="CALL",
                dte_min=5, dte_max=14, strike_width=5.0,
            ) if runner.chain_fetcher else None
            if not spread:
                print("Option data probe: FAILED - no tradable quoted spread")
                return 3
            long_leg, short_leg, long_contract, short_contract = spread
            print(
                "Option data probe: PASS | "
                f"{long_contract.symbol} {long_leg.bid:.2f}x{long_leg.ask:.2f} / "
                f"{short_contract.symbol} {short_leg.bid:.2f}x{short_leg.ask:.2f}"
            )
            print("Communication-only mode complete. No order was submitted.")
            return 0

        feature_engine = SessionWarmFeatureEngine(min_session_bars=args.min_session_bars)
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

            df = _warm_bars_frame(stock, args.symbol)
            try:
                f = feature_engine.compute(df)
            except ValueError as exc:
                print(f"[{datetime.now(eastern):%H:%M:%S ET}] warm-state unavailable: {exc}")
                time.sleep(args.poll_interval)
                continue

            if not f.pop("trade_ready"):
                route_ready = f.pop("route_readiness")
                session_bars = f.pop("session_bars")
                print(
                    f"[{datetime.now(eastern):%H:%M:%S ET}] "
                    f"session structure warming {session_bars}/{args.min_session_bars} | {route_ready}"
                )
                time.sleep(args.poll_interval)
                continue

            route_ready = f.pop("route_readiness")
            session_bars = f.pop("session_bars")
            account = trading.get_account()
            equity = float(account.equity)
            options_bp = getattr(account, "options_buying_power", None)
            buying_power = float(options_bp) if options_bp not in (None, "") else float(account.buying_power)
            runner.update_account(equity=equity, buying_power=buying_power)

            result = runner.evaluate_opportunity(symbol=args.symbol, **f)
            print(
                f"[{datetime.now(eastern):%H:%M:%S ET}] "
                f"price={f['price']:.2f} vwap={f['vwap']:.2f} session_bars={session_bars} "
                f"ready={route_ready} action={result['action']} details={result.get('details', {})}"
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
