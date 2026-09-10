"""Ranked vertical selection for V14.5.1.

The prior selector chose the nearest expiration and one approximately $5-wide
ATM vertical.  This selector enumerates multiple legal call/put debit verticals,
fetches one batched snapshot set when the SDK supports it, scores executable
quality, and returns the best expression of the already-approved direction.

It does not create direction; it only chooses a better option structure.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import math

from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus

try:
    from alpaca.data.requests import OptionSnapshotRequest
    _SNAPSHOT_OK = True
except Exception:  # pragma: no cover
    OptionSnapshotRequest = None
    _SNAPSHOT_OK = False

from strategy.option_chain_fetcher import ContractInfo
from strategy.v14_5_options_runner import SnapshotEnrichedOptionChainFetcher


class RankedVerticalSelector(SnapshotEnrichedOptionChainFetcher):
    """Enumerate and rank multiple verticals instead of taking the first ATM pair."""

    def __init__(self, trading_client, data_client, config: Optional[dict] = None):
        super().__init__(trading_client, data_client)
        self.cfg = config or {}
        self.last_ranked_candidates: List[dict] = []
        self.last_greeks_available = False

    @staticmethod
    def _ctype(value: Any) -> str:
        return str(value or "").lower().replace("contracttype.", "")

    def _candidate_pairs(
        self,
        symbol: str,
        underlying_price: float,
        side: str,
        dte_min: int,
        dte_max: int,
    ) -> List[Tuple[ContractInfo, ContractInfo]]:
        today = date.today()
        req = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date_gte=str(today + timedelta(days=dte_min)),
            expiration_date_lte=str(today + timedelta(days=dte_max)),
            status=AssetStatus.ACTIVE,
        )
        try:
            response = self.trading_client.get_option_contracts(req)
        except Exception:
            return []
        contracts = list(getattr(response, "option_contracts", None) or [])
        option_type = "call" if side.upper() == "CALL" else "put"
        contracts = [c for c in contracts if self._ctype(getattr(c, "type", "")) == option_type]
        by_exp: Dict[str, list] = {}
        for c in contracts:
            by_exp.setdefault(str(c.expiration_date), []).append(c)

        max_exp = int(self.cfg.get("vertical_selector_max_expirations", 3))
        min_width = float(self.cfg.get("vertical_selector_min_width", 2.5))
        max_width = float(self.cfg.get("vertical_selector_max_width", 10.0))
        max_pairs = int(self.cfg.get("vertical_selector_max_pairs", 24))
        pairs = []
        for exp in sorted(by_exp)[:max_exp]:
            items = sorted(by_exp[exp], key=lambda c: float(c.strike_price))
            # Restrict long strikes to a practical ATM neighborhood.  Snapshot
            # delta scoring later decides which structure is best.
            longs = sorted(items, key=lambda c: abs(float(c.strike_price) - underlying_price))[:8]
            for long_c in longs:
                ls = float(long_c.strike_price)
                if side.upper() == "CALL":
                    shorts = [c for c in items if float(c.strike_price) > ls]
                else:
                    shorts = [c for c in items if float(c.strike_price) < ls]
                shorts = sorted(shorts, key=lambda c: abs(abs(float(c.strike_price) - ls) - 5.0))
                for short_c in shorts[:5]:
                    ss = float(short_c.strike_price)
                    width = abs(ss - ls)
                    if not (min_width <= width <= max_width):
                        continue
                    pairs.append((
                        ContractInfo(long_c.symbol, ls, exp, option_type, symbol),
                        ContractInfo(short_c.symbol, ss, exp, option_type, symbol),
                    ))
                    if len(pairs) >= max_pairs:
                        return pairs
        return pairs

    def _snapshot_map(self, symbols: List[str]) -> Dict[str, Any]:
        if not symbols or not _SNAPSHOT_OK or not hasattr(self.data_client, "get_option_snapshot"):
            return {}
        try:
            return self.data_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=list(dict.fromkeys(symbols)))
            ) or {}
        except Exception:
            return {}

    def _score_pair(self, long_leg, short_leg, underlying_price: float, long_c, short_c) -> float:
        if not long_leg.is_valid or not short_leg.is_valid:
            return -1.0
        spread_bid = long_leg.bid - short_leg.ask
        spread_ask = long_leg.ask - short_leg.bid
        spread_mid = long_leg.mid - short_leg.mid
        if spread_mid <= 0 or spread_ask <= 0:
            return -1.0
        width_ratio = max(0.0, spread_ask - spread_bid) / spread_mid
        width_score = max(0.0, 1.0 - width_ratio / 0.20)

        ld = abs(float(getattr(long_leg, "delta", 0.0) or 0.0))
        sd = abs(float(getattr(short_leg, "delta", 0.0) or 0.0))
        target = float(self.cfg.get("option_delta_target_abs", 0.55))
        delta_fit = max(0.0, 1.0 - abs(ld - target) / 0.30) if ld > 0 else 0.35
        delta_sep = max(0.0, min(1.0, (ld - sd) / 0.20)) if ld > 0 and sd > 0 else 0.35

        liv = float(getattr(long_leg, "iv", 0.0) or 0.0)
        siv = float(getattr(short_leg, "iv", 0.0) or 0.0)
        iv_skew = abs(liv - siv) if liv > 0 and siv > 0 else 0.0
        iv_score = max(0.0, 1.0 - iv_skew / max(float(self.cfg.get("option_max_leg_iv_skew", 0.15)), 1e-9))

        width = abs(short_c.strike - long_c.strike)
        max_profit = max(0.0, width - spread_ask)
        rr = max_profit / spread_ask if spread_ask > 0 else 0.0
        payoff_score = max(0.0, min(1.0, rr / 2.0))

        # Prefer long strikes near the underlying but let delta and execution
        # quality dominate the final choice.
        atm_distance = abs(long_c.strike - underlying_price) / max(underlying_price, 1e-9)
        atm_score = max(0.0, 1.0 - atm_distance / 0.05)
        return (
            0.28 * width_score
            + 0.22 * delta_fit
            + 0.15 * delta_sep
            + 0.10 * iv_score
            + 0.15 * payoff_score
            + 0.10 * atm_score
        )

    def get_spread_with_quotes(
        self,
        symbol: str,
        underlying_price: float,
        side: str,
        dte_min: int = 5,
        dte_max: int = 14,
        strike_width: float = 5.0,
    ):
        pairs = self._candidate_pairs(symbol, underlying_price, side, dte_min, dte_max)
        if not pairs:
            return None

        snap_map = self._snapshot_map([c.symbol for pair in pairs for c in pair])
        ranked = []
        for long_c, short_c in pairs:
            legs = None
            if snap_map:
                lsn = snap_map.get(long_c.symbol)
                ssn = snap_map.get(short_c.symbol)
                if lsn is not None and ssn is not None:
                    long_leg = self._leg_from_snapshot(long_c, "buy", lsn)
                    short_leg = self._leg_from_snapshot(short_c, "sell", ssn)
                    if long_leg is not None and short_leg is not None:
                        legs = (long_leg, short_leg)
            if legs is None:
                legs = super().get_live_quotes(long_c, short_c)
            if not legs:
                continue
            long_leg, short_leg = legs
            score = self._score_pair(long_leg, short_leg, underlying_price, long_c, short_c)
            if score < 0:
                continue
            ranked.append((score, long_leg, short_leg, long_c, short_c))

        ranked.sort(key=lambda x: x[0], reverse=True)
        self.last_ranked_candidates = [
            {
                "score": x[0],
                "long": x[3].symbol,
                "short": x[4].symbol,
                "long_delta": float(getattr(x[1], "delta", 0.0) or 0.0),
                "short_delta": float(getattr(x[2], "delta", 0.0) or 0.0),
            }
            for x in ranked[:10]
        ]
        if not ranked or ranked[0][0] < float(self.cfg.get("vertical_selector_min_score", 0.48)):
            return None
        _, long_leg, short_leg, long_c, short_c = ranked[0]
        self.last_greeks_available = bool(
            abs(float(getattr(long_leg, "delta", 0.0) or 0.0)) > 0
            and abs(float(getattr(short_leg, "delta", 0.0) or 0.0)) > 0
        )
        return long_leg, short_leg, long_c, short_c
