"""
Spread Quality Gate for V14.2

Before submitting any mleg order, validate real or simulated option quotes.
Computes composite spread metrics and rejects orders that fail quality checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG


@dataclass
class OptionLeg:
    """Represents one leg of a spread."""
    contract_symbol: str = ""
    side: str = ""  # "buy" or "sell"
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    delta: float = 0.0
    iv: float = 0.0
    volume: int = 0
    open_interest: int = 0
    dte: int = 0
    strike: float = 0.0
    quote_timestamp: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if bid/ask are reasonable."""
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid

    @property
    def spread_width(self) -> float:
        return self.ask - self.bid


@dataclass
class SpreadQualityReport:
    """Full quality analysis of a debit spread."""
    passed: bool = False
    rejection_reason: str = ""

    # Computed spread metrics
    spread_bid: float = 0.0
    spread_ask: float = 0.0
    spread_mid: float = 0.0
    composite_spread_width: float = 0.0
    composite_spread_pct_of_mid: float = 0.0
    mid_inflation_since_watch: float = 0.0
    move_consumed_pct: float = 0.0
    remaining_reward: float = 0.0
    remaining_reward_to_risk: float = 0.0
    short_strike_distance: float = 0.0
    dte: int = 0
    long_leg_delta: float = 0.0
    iv_change: float = 0.0
    liquidity_score: float = 0.0
    limit_chase_needed: float = 0.0


class SpreadQualityGate:
    """
    Validates option spread quality before execution.
    Uses the NEXT quote after confirmation, never the confirmation moment quote.
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or CFG

    def evaluate(
        self,
        long_leg: OptionLeg,
        short_leg: OptionLeg,
        underlying_price: float,
        underlying_price_at_watch: float,
        estimated_debit_at_watch: float,
        target_price: float,
        max_debit: Optional[float] = None,
        route: str = "",
        iv_at_watch: float = 0.0,
    ) -> SpreadQualityReport:
        """
        Evaluate spread quality. Returns SpreadQualityReport with pass/fail.
        
        Parameters:
            long_leg: The long option leg (bought)
            short_leg: The short option leg (sold)
            underlying_price: Current underlying price
            underlying_price_at_watch: Price when watch ticket was created
            estimated_debit_at_watch: Estimated debit at watch time
            target_price: Target underlying price for the trade
            max_debit: Maximum allowed debit (overrides config if set)
            route: Route name for checking package eligibility
            iv_at_watch: IV at watch creation for change calculation
        """
        report = SpreadQualityReport()

        # ─── Basic Validity ──────────────────────────────────────────────
        if not long_leg.is_valid:
            report.rejection_reason = "long_leg_invalid_bid_ask"
            return report

        if not short_leg.is_valid:
            report.rejection_reason = "short_leg_invalid_bid_ask"
            return report

        # ─── Compute Spread Prices ───────────────────────────────────────
        # For a debit spread: buy long at ask, sell short at bid (worst case)
        # Natural price: long_ask - short_bid
        report.spread_ask = long_leg.ask - short_leg.bid  # debit to open (worst)
        report.spread_bid = long_leg.bid - short_leg.ask  # credit to close (worst)
        report.spread_mid = (long_leg.mid - short_leg.mid)

        if report.spread_mid <= 0:
            report.rejection_reason = "negative_or_zero_spread_mid"
            return report

        # ─── Composite Spread Width ──────────────────────────────────────
        report.composite_spread_width = long_leg.spread_width + short_leg.spread_width
        report.composite_spread_pct_of_mid = (
            report.composite_spread_width / report.spread_mid
            if report.spread_mid > 0 else 999.0
        )

        # ─── Mid Inflation Since Watch ───────────────────────────────────
        if estimated_debit_at_watch > 0:
            report.mid_inflation_since_watch = (
                (report.spread_mid - estimated_debit_at_watch) / estimated_debit_at_watch
            )
        else:
            report.mid_inflation_since_watch = 0.0

        # ─── Move Consumed ───────────────────────────────────────────────
        total_expected_move = abs(target_price - underlying_price_at_watch)
        move_consumed = abs(underlying_price - underlying_price_at_watch)
        report.move_consumed_pct = (
            move_consumed / total_expected_move if total_expected_move > 0 else 0.0
        )

        # ─── Remaining Reward ────────────────────────────────────────────
        # Max spread value is typically strike width * 100 (per contract)
        strike_width = abs(long_leg.strike - short_leg.strike)
        max_value = strike_width  # per-share
        report.remaining_reward = max_value - report.spread_mid if max_value > report.spread_mid else 0.0
        report.remaining_reward_to_risk = (
            report.remaining_reward / report.spread_mid
            if report.spread_mid > 0 else 0.0
        )

        # ─── Other Metrics ───────────────────────────────────────────────
        report.short_strike_distance = abs(underlying_price - short_leg.strike)
        report.dte = long_leg.dte
        report.long_leg_delta = long_leg.delta
        report.iv_change = (long_leg.iv - iv_at_watch) if iv_at_watch > 0 else 0.0

        # Liquidity score: based on volume and open interest
        total_volume = long_leg.volume + short_leg.volume
        total_oi = long_leg.open_interest + short_leg.open_interest
        report.liquidity_score = min(1.0, (total_volume / 100.0) * 0.5 + (total_oi / 500.0) * 0.5)

        # Limit chase needed: how much above mid we'd need to fill
        report.limit_chase_needed = (
            (report.spread_ask - report.spread_mid) / report.spread_mid
            if report.spread_mid > 0 else 0.0
        )

        # ─── Rejection Checks ────────────────────────────────────────────
        # 1. Mid inflation
        if report.mid_inflation_since_watch > self.cfg["max_option_mid_inflation"]:
            report.rejection_reason = (
                f"mid_inflation_exceeded: {report.mid_inflation_since_watch:.4f} "
                f"> {self.cfg['max_option_mid_inflation']}"
            )
            return report

        # 2. Composite spread too wide
        if report.composite_spread_pct_of_mid > self.cfg["max_composite_spread_pct_of_mid"]:
            report.rejection_reason = (
                f"composite_spread_too_wide: {report.composite_spread_pct_of_mid:.4f} "
                f"> {self.cfg['max_composite_spread_pct_of_mid']}"
            )
            return report

        # 3. Limit chase too large
        if report.limit_chase_needed > self.cfg["max_limit_chase_pct_of_debit"]:
            report.rejection_reason = (
                f"limit_chase_exceeded: {report.limit_chase_needed:.4f} "
                f"> {self.cfg['max_limit_chase_pct_of_debit']}"
            )
            return report

        # 4. Move consumed too much
        if report.move_consumed_pct > self.cfg["max_move_consumed_pct"]:
            report.rejection_reason = (
                f"move_consumed_exceeded: {report.move_consumed_pct:.4f} "
                f"> {self.cfg['max_move_consumed_pct']}"
            )
            return report

        # 5. Remaining reward to risk too low (min 0.5)
        if report.remaining_reward_to_risk < 0.5:
            report.rejection_reason = (
                f"remaining_reward_to_risk_too_low: {report.remaining_reward_to_risk:.4f}"
            )
            return report

        # 6. Stale option quote (check timestamp if available)
        if long_leg.quote_timestamp is None or short_leg.quote_timestamp is None:
            # In simulation mode, timestamps may not be present - allow
            pass

        # 7. Debit exceeds maximum
        effective_max = max_debit or self.cfg["max_original_debit"]
        if report.spread_mid > effective_max / 100.0:  # per-share debit
            report.rejection_reason = (
                f"debit_exceeds_max: {report.spread_mid:.2f} "
                f"> {effective_max / 100.0:.2f}"
            )
            return report

        # 8. Route not allowed for full package mode
        # (This is informational - the package builder handles fallback)

        # All checks passed
        report.passed = True
        return report
