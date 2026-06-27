"""
Route Expectancy Monitor for V14.2

Monitors per-route real profit factor and expectancy over rolling windows.
Implements adaptive rules:
- If route_real_profit_factor_50 < 1.5 → route becomes SOFT_SIZE_ONLY
- If route_real_profit_factor_100 < 1.0 → route becomes DISABLED
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RouteTradeRecord:
    """Record of a trade for route monitoring."""
    timestamp: datetime
    route: str
    pnl: float
    was_winner: bool


class RouteExpectancyMonitor:
    """
    Rolling route performance monitor with adaptive degradation.
    
    Adaptive rules:
    - profit_factor_50 < 1.5 → SOFT_SIZE_ONLY
    - profit_factor_100 < 1.0 → DISABLED
    """

    def __init__(self, log_dir: str = "HFT/logs/v14_2"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Rolling windows per route
        self._route_records_50: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._route_records_100: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Route status
        self._route_status: Dict[str, str] = {}  # ACTIVE, SOFT_SIZE_ONLY, DISABLED
        self._log_path = self.log_dir / "route_expectancy.csv"
        self._init_log()

    def _init_log(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "route", "pf_50", "pf_100",
                    "expectancy_50", "expectancy_100",
                    "win_rate_50", "status",
                ])

    def record_trade(self, route: str, pnl: float):
        """Record a trade result for a route."""
        record = RouteTradeRecord(
            timestamp=datetime.utcnow(),
            route=route,
            pnl=pnl,
            was_winner=pnl > 0,
        )
        self._route_records_50[route].append(record)
        self._route_records_100[route].append(record)
        self._evaluate_route(route)

    def get_route_status(self, route: str) -> str:
        """Get current route status: ACTIVE, SOFT_SIZE_ONLY, or DISABLED."""
        return self._route_status.get(route, "ACTIVE")

    def is_route_allowed(self, route: str) -> bool:
        """Check if route is allowed for trading."""
        status = self.get_route_status(route)
        return status != "DISABLED"

    def is_route_full_size(self, route: str) -> bool:
        """Check if route can use full position size."""
        status = self.get_route_status(route)
        return status == "ACTIVE"

    def get_profit_factor(self, route: str, window: int = 50) -> float:
        """Get rolling profit factor for a route."""
        records = (
            self._route_records_50[route] if window <= 50
            else self._route_records_100[route]
        )
        if not records:
            return float("inf")

        wins = sum(r.pnl for r in records if r.pnl > 0)
        losses = abs(sum(r.pnl for r in records if r.pnl <= 0))
        return wins / losses if losses > 0 else float("inf")

    def get_expectancy(self, route: str, window: int = 50) -> float:
        """Get rolling expectancy for a route."""
        records = (
            self._route_records_50[route] if window <= 50
            else self._route_records_100[route]
        )
        if not records:
            return 0.0
        return sum(r.pnl for r in records) / len(records)

    def _evaluate_route(self, route: str):
        """Evaluate route status based on adaptive rules."""
        pf_50 = self.get_profit_factor(route, 50)
        pf_100 = self.get_profit_factor(route, 100)
        exp_50 = self.get_expectancy(route, 50)
        exp_100 = self.get_expectancy(route, 100)

        # Only apply adaptive rules after sufficient data
        records_50 = self._route_records_50[route]
        records_100 = self._route_records_100[route]

        old_status = self._route_status.get(route, "ACTIVE")
        new_status = "ACTIVE"

        if len(records_100) >= 50 and pf_100 < 1.0:
            new_status = "DISABLED"
        elif len(records_50) >= 25 and pf_50 < 1.5:
            new_status = "SOFT_SIZE_ONLY"

        self._route_status[route] = new_status

        # Log if status changed
        if new_status != old_status:
            win_rate = (
                sum(1 for r in records_50 if r.was_winner) / len(records_50)
                if records_50 else 0.0
            )
            with open(self._log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.utcnow().isoformat(),
                    route, f"{pf_50:.3f}", f"{pf_100:.3f}",
                    f"{exp_50:.2f}", f"{exp_100:.2f}",
                    f"{win_rate:.3f}", new_status,
                ])

    def get_all_route_stats(self) -> Dict[str, dict]:
        """Get stats for all monitored routes."""
        stats = {}
        for route in set(list(self._route_records_50.keys()) + list(self._route_records_100.keys())):
            records_50 = self._route_records_50[route]
            stats[route] = {
                "status": self.get_route_status(route),
                "pf_50": self.get_profit_factor(route, 50),
                "pf_100": self.get_profit_factor(route, 100),
                "expectancy_50": self.get_expectancy(route, 50),
                "expectancy_100": self.get_expectancy(route, 100),
                "trades_50": len(records_50),
                "trades_100": len(self._route_records_100[route]),
            }
        return stats
