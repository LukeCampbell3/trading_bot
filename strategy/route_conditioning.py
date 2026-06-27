"""
Route Conditioning for V14.2

Evaluates underlying price action to determine which route(s) are active
and computes route scores for watch ticket admission.

Routes:
- VWAP_PULLBACK: Price pulls back to VWAP in a trending day, then reclaims
- PULLBACK_CONTINUATION: Trend continuation after orderly pullback
- DOMINANT_TREND_PULLBACK: Strong trend with shallow retracement
- PUT_REJECTION: Price rejects from VWAP/resistance, downside setup
- RAW_BREAKOUT (soft-only): Breakout without confirmed continuation
- LATE_BREAKOUT (soft-only): Breakout entry late in the move
- NEWS_SPIKE (soft-only): Spike from news catalyst
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG


@dataclass
class RouteCandidate:
    """A scored route candidate for a symbol."""
    route: str = ""
    side: str = "CALL"  # "CALL" or "PUT"
    score: float = 0.0
    ic_spread: float = 0.0
    ev_over_debit: float = 0.0
    option_liquidity: float = 0.0
    expected_move: float = 0.0
    reason: str = ""


class RouteConditioner:
    """
    Analyzes price action to determine which routes are active and their scores.
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or CFG

    def evaluate_routes(
        self,
        price: float,
        vwap: float,
        atr: float,
        high_of_day: float,
        low_of_day: float,
        trend_slope: float,
        volume_ratio: float,
        price_5m_ago: float,
        price_15m_ago: float,
        option_liquidity: float = 0.7,
        iv_percentile: float = 0.5,
    ) -> List[RouteCandidate]:
        """
        Score all routes based on current market conditions.
        Returns sorted list of qualifying candidates (score >= watch_min).
        """
        candidates = []

        if atr <= 0 or price <= 0:
            return candidates

        # Compute shared metrics
        price_vs_vwap = (price - vwap) / atr if atr > 0 else 0.0
        day_range = high_of_day - low_of_day
        day_range_atr = day_range / atr if atr > 0 else 0.0
        ret_5m = (price - price_5m_ago) / price_5m_ago if price_5m_ago > 0 else 0.0
        ret_15m = (price - price_15m_ago) / price_15m_ago if price_15m_ago > 0 else 0.0

        # ─── CALL Routes ─────────────────────────────────────────────────
        # VWAP_PULLBACK: trending up day, price near VWAP, about to reclaim
        vwap_pb_score = self._score_vwap_pullback(
            price_vs_vwap, trend_slope, volume_ratio, day_range_atr, ret_5m
        )
        if vwap_pb_score > 0:
            candidates.append(RouteCandidate(
                route="VWAP_PULLBACK", side="CALL", score=vwap_pb_score,
                option_liquidity=option_liquidity,
                expected_move=atr * 1.5,
            ))

        # PULLBACK_CONTINUATION
        pb_cont_score = self._score_pullback_continuation(
            price_vs_vwap, trend_slope, ret_5m, ret_15m, volume_ratio
        )
        if pb_cont_score > 0:
            candidates.append(RouteCandidate(
                route="PULLBACK_CONTINUATION", side="CALL", score=pb_cont_score,
                option_liquidity=option_liquidity,
                expected_move=atr * 2.0,
            ))

        # DOMINANT_TREND_PULLBACK
        dom_score = self._score_dominant_trend(
            price_vs_vwap, trend_slope, day_range_atr, ret_15m
        )
        if dom_score > 0:
            candidates.append(RouteCandidate(
                route="DOMINANT_TREND_PULLBACK", side="CALL", score=dom_score,
                option_liquidity=option_liquidity,
                expected_move=atr * 2.5,
            ))

        # ─── PUT Routes ──────────────────────────────────────────────────
        # PUT_REJECTION
        put_rej_score = self._score_put_rejection(
            price_vs_vwap, trend_slope, ret_5m, volume_ratio
        )
        if put_rej_score > 0:
            candidates.append(RouteCandidate(
                route="PUT_REJECTION", side="PUT", score=put_rej_score,
                option_liquidity=option_liquidity,
                expected_move=atr * 1.5,
            ))

        # ─── Soft-Only Routes ────────────────────────────────────────────
        # RAW_BREAKOUT
        raw_bo_score = self._score_raw_breakout(
            price, high_of_day, atr, volume_ratio, trend_slope
        )
        if raw_bo_score > 0:
            candidates.append(RouteCandidate(
                route="RAW_BREAKOUT", side="CALL", score=raw_bo_score,
                option_liquidity=option_liquidity,
                expected_move=atr * 1.0,
            ))

        # Compute IC spread and EV for each candidate
        for c in candidates:
            c.ic_spread = self._estimate_ic_spread(c.score, iv_percentile)
            c.ev_over_debit = self._estimate_ev_over_debit(c.score, c.expected_move, atr)

        # Filter by minimum watch score
        candidates = [c for c in candidates if c.score >= self.cfg["watch_min_route_score"]]
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _score_vwap_pullback(
        self, price_vs_vwap: float, trend: float, vol_ratio: float,
        day_range_atr: float, ret_5m: float
    ) -> float:
        """VWAP pullback: trending day, price near VWAP, reclaiming."""
        score = 0.0
        # Price should be near VWAP (within 0.5 ATR) and trending up
        if -0.5 <= price_vs_vwap <= 0.3 and trend > 0:
            score += 0.3
            # Stronger trend bonus
            score += min(0.2, trend * 50)
            # Volume supporting
            if vol_ratio > 1.0:
                score += min(0.15, (vol_ratio - 1.0) * 0.1)
            # Day range reasonable (not exhausted)
            if 0.5 < day_range_atr < 2.5:
                score += 0.15
            # Recent 5m move showing reclaim
            if ret_5m > 0:
                score += min(0.1, ret_5m * 100)
        return min(1.0, score)

    def _score_pullback_continuation(
        self, price_vs_vwap: float, trend: float, ret_5m: float,
        ret_15m: float, vol_ratio: float
    ) -> float:
        """Pullback continuation: strong trend with recent pullback resolving."""
        score = 0.0
        if price_vs_vwap > 0.2 and trend > 0.001:
            score += 0.25
            if ret_15m > 0 and ret_5m > 0:
                score += 0.2
            if trend > 0.003:
                score += 0.2
            if vol_ratio > 0.8:
                score += 0.1
            # Pullback evidence: 5m was negative recently but now positive
            if ret_5m > 0 and ret_5m < ret_15m * 0.5:
                score += 0.15
        return min(1.0, score)

    def _score_dominant_trend(
        self, price_vs_vwap: float, trend: float, day_range_atr: float,
        ret_15m: float
    ) -> float:
        """Dominant trend: very strong directional day with shallow retrace."""
        score = 0.0
        if price_vs_vwap > 0.5 and trend > 0.003 and day_range_atr > 1.5:
            score += 0.35
            if trend > 0.005:
                score += 0.2
            if ret_15m > 0.003:
                score += 0.15
            # Shallow retrace = price still well above VWAP
            if price_vs_vwap > 1.0:
                score += 0.15
        return min(1.0, score)

    def _score_put_rejection(
        self, price_vs_vwap: float, trend: float, ret_5m: float,
        vol_ratio: float
    ) -> float:
        """Put rejection: price fails at VWAP from below, or rejects resistance."""
        score = 0.0
        if price_vs_vwap < -0.2 and trend < 0:
            score += 0.25
            if ret_5m < -0.001:
                score += 0.2
            if abs(trend) > 0.002:
                score += 0.15
            if vol_ratio > 1.0:
                score += 0.1
            # VWAP rejection: was near VWAP and fell away
            if -0.8 < price_vs_vwap < -0.2:
                score += 0.15
        return min(1.0, score)

    def _score_raw_breakout(
        self, price: float, high_of_day: float, atr: float,
        vol_ratio: float, trend: float
    ) -> float:
        """Raw breakout: price near/above HOD with volume (soft-only)."""
        score = 0.0
        if atr > 0 and price > 0:
            dist_to_hod = (high_of_day - price) / atr
            if dist_to_hod < 0.2 and trend > 0:  # Near or above HOD
                score += 0.3
                if vol_ratio > 1.5:
                    score += 0.2
                if trend > 0.003:
                    score += 0.15
        return min(1.0, score)

    def _estimate_ic_spread(self, route_score: float, iv_percentile: float) -> float:
        """Estimate information coefficient spread based on route quality."""
        base_ic = route_score * 0.05
        iv_adjustment = (0.5 - iv_percentile) * 0.01  # lower IV = tighter IC
        return max(0.0, base_ic + iv_adjustment)

    def _estimate_ev_over_debit(self, route_score: float, expected_move: float, atr: float) -> float:
        """Estimate expected value over debit ratio."""
        if atr <= 0:
            return 0.0
        move_quality = expected_move / atr
        return route_score * move_quality * 0.08  # calibrated estimate
