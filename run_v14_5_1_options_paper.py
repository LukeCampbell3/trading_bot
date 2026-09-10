"""Paper entrypoint for V14.5.1 validation-hardened options trading."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import pytz

from run_v14_4_options_paper import _make_clients, _warm_bars_frame
from strategy.v14_3_highvol_config import can_trade_symbol
from strategy.v14_5_1_features import NormalizedSessionWarmFeatureEngine
from strategy.v14_5_1_hardened_options import create_v14_5_1_options_trader


def main() -> int:
    parser = argparse.ArgumentParser(description="V14.5.1 validation-hardened Alpaca paper options")
    parser.add_argument("--symbol", default="COIN", choices=["COIN", "TSLA"])
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--communication-only", action="store_true")
    parser.add_argument("--min-session-bars", type=int, default=5)
    args = parser.parse_args()

    allowed, reason, size = can_trade_symbol(args.symbol)
    if not allowed:
        print(f"BLOCKED by symbol policy: {reason}")
        return 2

    print("=" * 92)
    print("V14.5.1 VALIDATION HARDENING + V14.4 ZERO-WAIT WARM START")
    print(f"symbol={args.symbol} base_symbol_size={size:.2f}")
    print("instrument=CALL/PUT DEBIT SPREADS | order=LIMIT MLEG | LIVE TRADING=DISABLED")
    print("trend=normalized fraction/min | IC/EV=empirical labels | partial fills=reconciled")
    print("=" * 92)

    try:
        trading, stock, options = _make_clients()
        account = trading.get_account()
        print(f"Trading API: connected | status={getattr(account, 'status', '')}")
        print(f"options_approved_level={getattr(account, 'options_approved_level', None)}")
        print(f"options_trading_level={getattr(account, 'options_trading_level', None)}")

        trader = create_v14_5_1_options_trader(
            args.symbol,
            trading_client=trading,
            option_data_client=options,
            paper_mode=True,
        )
        probe = trader.execution_manager.communication_probe()
        print(f"Broker probe: {probe}")

        df = _warm_bars_frame(stock, args.symbol)
        if df.empty:
            print("Stock data probe: FAILED")
            return 3

        if args.communication_only:
            px = float(df["close"].iloc[-1])
            spread = trader.chain_fetcher.get_spread_with_quotes(
                args.symbol,
                underlying_price=px,
                side="CALL",
                dte_min=5,
                dte_max=14,
                strike_width=5.0,
            ) if trader.chain_fetcher else None
            if not spread:
                print("Option data probe: FAILED - no ranked tradable spread")
                return 3
            long_leg, short_leg, long_contract, short_contract = spread
            print(
                "Option data probe: PASS | "
                f"{long_contract.symbol} {long_leg.bid:.2f}x{long_leg.ask:.2f} "
                f"delta={long_leg.delta:.3f} iv={long_leg.iv:.3f} / "
                f"{short_contract.symbol} {short_leg.bid:.2f}x{short_leg.ask:.2f} "
                f"delta={short_leg.delta:.3f} iv={short_leg.iv:.3f}"
            )
            print(f"Top selector candidates: {trader.chain_fetcher.last_ranked_candidates[:5]}")
            print("Communication-only complete. No order submitted.")
            return 0

        feature_engine = NormalizedSessionWarmFeatureEngine(min_session_bars=args.min_session_bars)
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
                features = feature_engine.compute(df)
            except ValueError as exc:
                print(f"[{datetime.now(eastern):%H:%M:%S ET}] warm-state unavailable: {exc}")
                time.sleep(args.poll_interval)
                continue

            trade_ready = features.pop("trade_ready")
            route_ready = features.pop("route_readiness")
            session_bars = features.pop("session_bars")
            raw_trend = features.pop("trend_slope_raw_dollars_per_min", 0.0)
            features.pop("trend_units", None)
            if not trade_ready:
                print(
                    f"[{datetime.now(eastern):%H:%M:%S ET}] "
                    f"session structure warming {session_bars}/{args.min_session_bars} | {route_ready}"
                )
                time.sleep(args.poll_interval)
                continue

            account = trading.get_account()
            equity = float(account.equity)
            options_bp = getattr(account, "options_buying_power", None)
            buying_power = float(options_bp) if options_bp not in (None, "") else float(account.buying_power)
            trader.update_account(equity=equity, buying_power=buying_power)

            market_timestamp = df.index[-1].isoformat() if len(df.index) else None
            result = trader.evaluate_opportunity(
                symbol=args.symbol,
                market_timestamp=market_timestamp,
                **features,
            )
            status = trader.get_status()
            print(
                f"[{datetime.now(eastern):%H:%M:%S ET}] "
                f"price={features['price']:.2f} vwap={features['vwap']:.2f} "
                f"trend={features['trend_slope']:.6f}/min raw=${raw_trend:.4f}/min "
                f"bias={status['direction_bias']} flip_streak={status['reversal_flip_streak']} "
                f"action={result['action']} emergency={status['emergency_exit_required']}"
            )
            time.sleep(args.poll_interval)

        print(trader.get_status())
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
