"""Read-only Alpaca options communication test for V14.3.

No orders are submitted. The script proves only that credentials, the paper
Trading API, option contract discovery, and current option quotes are reachable.
It also verifies that the installed alpaca-py can serialize an MLEG limit order.

Usage:
    python test_alpaca_options_connection.py --symbol COIN
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from alpaca_config import AlpacaConfig
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from strategy.option_chain_fetcher import OptionChainFetcher
from strategy.v14_3_highvol_config import can_trade_symbol


def latest_underlying_price(stock_client, symbol: str) -> float:
    now = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=now - timedelta(days=2),
        end=now,
        feed=DataFeed.IEX,
        limit=1000,
    )
    bars = stock_client.get_stock_bars(req)
    values = bars[symbol]
    if not values:
        raise RuntimeError(f"no stock bars returned for {symbol}")
    return float(values[-1].close)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="COIN", choices=["COIN", "TSLA"])
    args = parser.parse_args()
    symbol = args.symbol.upper()

    allowed, reason, size = can_trade_symbol(symbol)
    print(f"V14.3 policy: allowed={allowed} reason={reason} size_multiplier={size}")
    if not allowed:
        return 2

    checks = []
    try:
        AlpacaConfig.validate()
        checks.append(("credentials", True, "loaded"))
    except Exception as exc:
        checks.append(("credentials", False, str(exc)))
        _print(checks)
        return 1

    try:
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
        option_data = OptionHistoricalDataClient(
            AlpacaConfig.API_KEY,
            AlpacaConfig.API_SECRET,
        )
        account = trading.get_account()
        clock = trading.get_clock()
        checks.append((
            "paper_trading_api",
            True,
            f"status={getattr(account, 'status', '')} market_open={getattr(clock, 'is_open', False)}",
        ))
        checks.append((
            "options_permission",
            (getattr(account, "options_trading_level", 0) or 0) >= 3,
            f"approved={getattr(account, 'options_approved_level', None)} trading={getattr(account, 'options_trading_level', None)}",
        ))
    except Exception as exc:
        checks.append(("paper_trading_api", False, str(exc)))
        _print(checks)
        return 1

    try:
        underlying = latest_underlying_price(stock, symbol)
        checks.append(("stock_market_data", True, f"{symbol}={underlying:.2f}"))
    except Exception as exc:
        checks.append(("stock_market_data", False, str(exc)))
        _print(checks)
        return 1

    try:
        fetcher = OptionChainFetcher(trading, option_data)
        spread = fetcher.get_spread_with_quotes(
            symbol=symbol,
            underlying_price=underlying,
            side="CALL",
            dte_min=5,
            dte_max=14,
            strike_width=5.0,
        )
        if not spread:
            raise RuntimeError("no quoted call debit spread returned")
        long_leg, short_leg, long_contract, short_contract = spread
        if not long_leg.is_valid or not short_leg.is_valid:
            raise RuntimeError("invalid bid/ask on selected option legs")
        checks.append((
            "option_contracts_and_quotes",
            True,
            f"{long_contract.symbol} {long_leg.bid:.2f}x{long_leg.ask:.2f}; "
            f"{short_contract.symbol} {short_leg.bid:.2f}x{short_leg.ask:.2f}",
        ))
    except Exception as exc:
        checks.append(("option_contracts_and_quotes", False, str(exc)))
        _print(checks)
        return 1

    # SDK serialization only. DO NOT submit this request.
    try:
        dry_run = LimitOrderRequest(
            qty=1,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=1.00,
            legs=[
                OptionLegRequest(
                    symbol=long_contract.symbol,
                    ratio_qty=1.0,
                    side=OrderSide.BUY,
                    position_intent=PositionIntent.BUY_TO_OPEN,
                ),
                OptionLegRequest(
                    symbol=short_contract.symbol,
                    ratio_qty=1.0,
                    side=OrderSide.SELL,
                    position_intent=PositionIntent.SELL_TO_OPEN,
                ),
            ],
            client_order_id="v14_3_dry_run_only",
        )
        payload = dry_run.to_request_fields()
        checks.append((
            "mleg_request_serialization",
            payload.get("order_class") in ("mleg", OrderClass.MLEG),
            "serialized; NOT SUBMITTED",
        ))
    except Exception as exc:
        checks.append(("mleg_request_serialization", False, str(exc)))

    _print(checks)
    return 0 if all(ok for _, ok, _ in checks) else 1


def _print(checks):
    print("\n" + "=" * 72)
    print("ALPACA OPTIONS COMMUNICATION - READ ONLY")
    print("=" * 72)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4} | {name:30} | {detail}")
    print("No order was submitted.")


if __name__ == "__main__":
    sys.exit(main())
