"""
Option Chain Fetcher for V14.2

Fetches real option contracts and quotes from Alpaca API.
Selects appropriate strikes for debit spread construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, List, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

from strategy.spread_quality_gate import OptionLeg


@dataclass
class ContractInfo:
    """Resolved option contract from Alpaca."""
    symbol: str
    strike: float
    expiration: str
    option_type: str  # "call" or "put"
    underlying_symbol: str


class OptionChainFetcher:
    """
    Fetches real option contracts and live quotes from Alpaca.
    Selects ATM/OTM strikes suitable for debit spreads.
    """

    def __init__(self, trading_client: TradingClient, data_client: OptionHistoricalDataClient):
        self.trading_client = trading_client
        self.data_client = data_client

    def get_spread_contracts(
        self,
        symbol: str,
        underlying_price: float,
        side: str,  # "CALL" or "PUT"
        dte_min: int = 5,
        dte_max: int = 14,
        strike_width: float = 5.0,
    ) -> Optional[Tuple[ContractInfo, ContractInfo]]:
        """
        Find suitable long and short contracts for a debit spread.

        For CALL debit spread: buy ATM/slightly-ITM call, sell OTM call
        For PUT debit spread: buy ATM/slightly-ITM put, sell OTM put

        Returns (long_contract, short_contract) or None if not found.
        """
        today = date.today()
        exp_after = today + timedelta(days=dte_min)
        exp_before = today + timedelta(days=dte_max)

        option_type = "call" if side == "CALL" else "put"

        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                expiration_date_gte=str(exp_after),
                expiration_date_lte=str(exp_before),
                status=AssetStatus.ACTIVE,
            )
            response = self.trading_client.get_option_contracts(req)
        except Exception as e:
            print(f"[CHAIN] Failed to fetch contracts: {e}")
            return None

        if not response or not hasattr(response, "option_contracts"):
            return None

        all_contracts = response.option_contracts or []
        if not all_contracts:
            return None

        # Filter by type
        contracts = [
            c for c in all_contracts
            if str(getattr(c, "type", "")).lower().replace("contracttype.", "") == option_type
        ]
        if not contracts:
            return None

        # Sort by expiration (prefer nearest), then by strike distance from ATM
        # Group by expiration
        by_exp = {}
        for c in contracts:
            exp = str(c.expiration_date)
            by_exp.setdefault(exp, []).append(c)

        # Pick the nearest valid expiration
        sorted_exps = sorted(by_exp.keys())
        if not sorted_exps:
            return None

        target_exp = sorted_exps[0]
        exp_contracts = by_exp[target_exp]

        # Sort by strike
        exp_contracts.sort(key=lambda c: float(c.strike_price))

        # Find ATM strike (closest to underlying price)
        atm_idx = min(
            range(len(exp_contracts)),
            key=lambda i: abs(float(exp_contracts[i].strike_price) - underlying_price)
        )

        if side == "CALL":
            # Long: ATM or slightly ITM (strike <= price)
            # Short: OTM (strike > price, at least strike_width away from long)
            long_contract = exp_contracts[atm_idx]
            long_strike = float(long_contract.strike_price)

            # Find short strike approximately strike_width above long
            short_contract = None
            for c in exp_contracts[atm_idx + 1:]:
                if float(c.strike_price) >= long_strike + strike_width * 0.8:
                    short_contract = c
                    break

            if short_contract is None:
                # Try with smaller width
                for c in exp_contracts[atm_idx + 1:]:
                    if float(c.strike_price) > long_strike:
                        short_contract = c
                        break

        else:  # PUT
            # Long: ATM or slightly ITM (strike >= price)
            # Short: OTM (strike < price, at least strike_width below long)
            long_contract = exp_contracts[atm_idx]
            long_strike = float(long_contract.strike_price)

            # Find short strike approximately strike_width below long
            short_contract = None
            for c in reversed(exp_contracts[:atm_idx]):
                if float(c.strike_price) <= long_strike - strike_width * 0.8:
                    short_contract = c
                    break

            if short_contract is None:
                for c in reversed(exp_contracts[:atm_idx]):
                    if float(c.strike_price) < long_strike:
                        short_contract = c
                        break

        if short_contract is None:
            return None

        long_info = ContractInfo(
            symbol=long_contract.symbol,
            strike=float(long_contract.strike_price),
            expiration=str(long_contract.expiration_date),
            option_type=option_type,
            underlying_symbol=symbol,
        )
        short_info = ContractInfo(
            symbol=short_contract.symbol,
            strike=float(short_contract.strike_price),
            expiration=str(short_contract.expiration_date),
            option_type=option_type,
            underlying_symbol=symbol,
        )

        return (long_info, short_info)

    def get_live_quotes(
        self, long_contract: ContractInfo, short_contract: ContractInfo
    ) -> Optional[Tuple[OptionLeg, OptionLeg]]:
        """
        Fetch live bid/ask quotes for both legs of a spread.
        Returns (long_leg, short_leg) with populated quote data.
        """
        symbols = [long_contract.symbol, short_contract.symbol]

        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_option_latest_quote(req)
        except Exception as e:
            print(f"[CHAIN] Failed to fetch quotes: {e}")
            return None

        if not quotes:
            return None

        long_quote = quotes.get(long_contract.symbol)
        short_quote = quotes.get(short_contract.symbol)

        if not long_quote or not short_quote:
            return None

        long_bid = float(long_quote.bid_price) if long_quote.bid_price else 0.0
        long_ask = float(long_quote.ask_price) if long_quote.ask_price else 0.0
        short_bid = float(short_quote.bid_price) if short_quote.bid_price else 0.0
        short_ask = float(short_quote.ask_price) if short_quote.ask_price else 0.0

        long_leg = OptionLeg(
            contract_symbol=long_contract.symbol,
            side="buy",
            bid=long_bid,
            ask=long_ask,
            mid=(long_bid + long_ask) / 2 if (long_bid + long_ask) > 0 else 0.0,
            strike=long_contract.strike,
            dte=(date.fromisoformat(long_contract.expiration) - date.today()).days,
            quote_timestamp=str(getattr(long_quote, "timestamp", "")),
        )
        short_leg = OptionLeg(
            contract_symbol=short_contract.symbol,
            side="sell",
            bid=short_bid,
            ask=short_ask,
            mid=(short_bid + short_ask) / 2 if (short_bid + short_ask) > 0 else 0.0,
            strike=short_contract.strike,
            dte=(date.fromisoformat(short_contract.expiration) - date.today()).days,
            quote_timestamp=str(getattr(short_quote, "timestamp", "")),
        )

        return (long_leg, short_leg)

    def get_spread_with_quotes(
        self,
        symbol: str,
        underlying_price: float,
        side: str,
        dte_min: int = 5,
        dte_max: int = 14,
        strike_width: float = 5.0,
    ) -> Optional[Tuple[OptionLeg, OptionLeg, ContractInfo, ContractInfo]]:
        """
        Combined: find contracts and fetch quotes in one call.
        Returns (long_leg, short_leg, long_contract, short_contract) or None.
        """
        contracts = self.get_spread_contracts(
            symbol, underlying_price, side, dte_min, dte_max, strike_width
        )
        if not contracts:
            return None

        long_contract, short_contract = contracts
        legs = self.get_live_quotes(long_contract, short_contract)
        if not legs:
            return None

        long_leg, short_leg = legs
        return (long_leg, short_leg, long_contract, short_contract)
