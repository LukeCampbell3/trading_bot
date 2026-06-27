"""
Fill Quality Logger for V14.2

Tracks and reports on fill quality metrics across all executions.
Used by telemetry monitors to detect degradation.
"""

from __future__ import annotations

import csv
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class FillRecord:
    """Single fill event for quality tracking."""
    timestamp: datetime
    order_id: str
    ticket_id: str
    direction: str  # "OPEN" or "CLOSE"
    intended_limit: float
    actual_price: float
    composite_mid: float
    composite_bid: float
    composite_ask: float
    slippage_vs_mid: float
    slippage_vs_bid: float
    time_to_fill_seconds: float
    was_missed: bool = False
    missed_reason: str = ""


class FillQualityLogger:
    """
    Aggregates fill quality data and provides rolling statistics.
    """

    def __init__(self, log_dir: str = "HFT/logs/v14_2", window: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._window = window
        self._records: deque = deque(maxlen=window * 2)
        self._entry_slippages: deque = deque(maxlen=window)
        self._exit_slippages: deque = deque(maxlen=window)
        self._missed_fills: deque = deque(maxlen=window)
        self._fill_times: deque = deque(maxlen=window)

    def record_fill(
        self,
        order_id: str,
        ticket_id: str,
        direction: str,
        intended_limit: float,
        actual_price: float,
        composite_mid: float,
        composite_bid: float,
        composite_ask: float,
        time_to_fill: float,
    ):
        """Record a successful fill."""
        slip_mid = (actual_price - composite_mid) / composite_mid if composite_mid > 0 else 0.0
        slip_bid = (composite_bid - actual_price) / composite_bid if composite_bid > 0 else 0.0

        record = FillRecord(
            timestamp=datetime.utcnow(),
            order_id=order_id,
            ticket_id=ticket_id,
            direction=direction,
            intended_limit=intended_limit,
            actual_price=actual_price,
            composite_mid=composite_mid,
            composite_bid=composite_bid,
            composite_ask=composite_ask,
            slippage_vs_mid=slip_mid,
            slippage_vs_bid=slip_bid,
            time_to_fill_seconds=time_to_fill,
        )
        self._records.append(record)

        if direction == "OPEN":
            self._entry_slippages.append(slip_mid)
        else:
            self._exit_slippages.append(slip_bid)
        self._fill_times.append(time_to_fill)
        self._missed_fills.append(False)

    def record_missed_fill(self, order_id: str, ticket_id: str, reason: str):
        """Record a missed fill event."""
        record = FillRecord(
            timestamp=datetime.utcnow(),
            order_id=order_id,
            ticket_id=ticket_id,
            direction="OPEN",
            intended_limit=0.0,
            actual_price=0.0,
            composite_mid=0.0,
            composite_bid=0.0,
            composite_ask=0.0,
            slippage_vs_mid=0.0,
            slippage_vs_bid=0.0,
            time_to_fill_seconds=0.0,
            was_missed=True,
            missed_reason=reason,
        )
        self._records.append(record)
        self._missed_fills.append(True)

    @property
    def entry_slippage_mean(self) -> float:
        if not self._entry_slippages:
            return 0.0
        return statistics.mean(self._entry_slippages)

    @property
    def exit_slippage_mean(self) -> float:
        if not self._exit_slippages:
            return 0.0
        return statistics.mean(self._exit_slippages)

    @property
    def missed_fill_rate(self) -> float:
        if not self._missed_fills:
            return 0.0
        return sum(1 for m in self._missed_fills if m) / len(self._missed_fills)

    @property
    def mean_fill_time(self) -> float:
        if not self._fill_times:
            return 0.0
        return statistics.mean(self._fill_times)

    def get_summary(self) -> dict:
        """Get rolling fill quality summary."""
        return {
            "entry_slippage_mean": self.entry_slippage_mean,
            "exit_slippage_mean": self.exit_slippage_mean,
            "missed_fill_rate": self.missed_fill_rate,
            "mean_fill_time_seconds": self.mean_fill_time,
            "total_records": len(self._records),
            "total_missed": sum(1 for r in self._records if r.was_missed),
        }
