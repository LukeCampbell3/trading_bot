"""V14.5.1 session features with dimensionless trend slope."""

from __future__ import annotations

from strategy.warm_start_v14_4 import SessionWarmFeatureEngine


class NormalizedSessionWarmFeatureEngine(SessionWarmFeatureEngine):
    """Keep V14.4 session isolation while removing price-scale dependence.

    SessionWarmFeatureEngine returns a blended OLS slope in dollars per minute.
    V14 route thresholds are dimensionless fractions, so V14.5.1 converts the
    slope to fraction-of-current-price per minute before route scoring.
    """

    def compute(self, df):
        out = super().compute(df)
        raw = float(out.get("trend_slope", 0.0) or 0.0)
        price = float(out.get("price", 0.0) or 0.0)
        out["trend_slope_raw_dollars_per_min"] = raw
        out["trend_slope"] = raw / price if price > 0 else 0.0
        out["trend_units"] = "fraction_per_minute"
        return out
