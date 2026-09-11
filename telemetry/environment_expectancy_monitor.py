"""
Environment Expectancy Monitor for V14.2

Monitors environment conditions and blocks entries when environment expectancy is negative.

Adaptive rule:
- If poor_environment_expectancy < 0, block new entries in that environment.
- If same-underlying losses exceed limits, disable that symbol for the week.
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EnvironmentState:
    """Market environment classification."""
    timestamp: datetime
    vix_level: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    trend_state: str = "NEUTRAL"  # STRONG_UP, UP, NEUTRAL, DOWN, STRONG_DOWN
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH
    correlation_state: str = "NORMAL"  # DECORRELATED, NORMAL, CORRELATED
    time_of_day: str = "MID"  # OPEN, EARLY, MID, LATE, CLOSE

    @property
    def label(self) -> str:
        return f"{self.vix_level}_{self.trend_state}_{self.volatility_regime}"


@dataclass
class EnvironmentTradeRecord:
    """Trade record with environment context."""
    timestamp: datetime
    environment_label: str
    symbol: str
    pnl: float


class EnvironmentExpectancyMonitor:
    """
    Tracks expectancy per environment classification.
    Blocks entries when environment has negative expectancy.
    """

    def __init__(self, log_dir: str = "HFT/logs/v14_2", window: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._window = window

        # Per-environment tracking
        self._env_records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._env_status: Dict[str, str] = {}  # ALLOWED, BLOCKED

        # Per-symbol tracking
        self._symbol_weekly_losses: Dict[str, float] = defaultdict(float)
        self._symbol_weekly_trades: Dict[str, int] = defaultdict(int)
        self._disabled_symbols: List[str] = []
        self._week_start: Optional[date] = None

        # After-loss tracking
        self._recent_loss_streak: int = 0
        self._after_loss_records: deque = deque(maxlen=window)

        self._log_path = self.log_dir / "environment_expectancy.csv"
        self._init_log()

    def _init_log(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "environment", "expectancy",
                    "trades", "status", "event",
                ])

    def record_trade(self, environment: EnvironmentState, symbol: str, pnl: float):
        """Record a trade result in the current environment."""
        label = environment.label
        record = EnvironmentTradeRecord(
            timestamp=datetime.utcnow(),
            environment_label=label,
            symbol=symbol,
            pnl=pnl,
        )
        self._env_records[label].append(record)

        # Track symbol losses
        self._check_week_rollover()
        if pnl < 0:
            self._symbol_weekly_losses[symbol] += abs(pnl)
            self._recent_loss_streak += 1
        else:
            self._recent_loss_streak = 0

        self._symbol_weekly_trades[symbol] += 1

        # After-loss tracking
        if self._recent_loss_streak > 0:
            self._after_loss_records.append(record)

        # Evaluate environment
        self._evaluate_environment(label)

    def is_environment_allowed(self, environment: EnvironmentState) -> bool:
        """Check if current environment allows new entries."""
        label = environment.label
        return self._env_status.get(label, "ALLOWED") == "ALLOWED"

    def is_symbol_allowed(self, symbol: str) -> bool:
        """Check if symbol is allowed (not disabled due to losses)."""
        return symbol not in self._disabled_symbols

    def get_environment_status(self, environment: EnvironmentState) -> str:
        """Get raw status ("ALLOWED" or "BLOCKED") for an environment label."""
        return self._env_status.get(environment.label, "ALLOWED")

    def get_environment_expectancy(self, environment: EnvironmentState) -> float:
        """Get rolling expectancy for an environment."""
        label = environment.label
        records = self._env_records.get(label)
        if not records or len(records) < 5:
            return 0.0  # Insufficient data, allow
        return sum(r.pnl for r in records) / len(records)

    def get_after_loss_expectancy(self) -> float:
        """Get expectancy of trades taken after a loss."""
        if not self._after_loss_records or len(self._after_loss_records) < 5:
            return 0.0
        return sum(r.pnl for r in self._after_loss_records) / len(self._after_loss_records)

    def _evaluate_environment(self, label: str):
        """Evaluate and potentially block an environment."""
        records = self._env_records[label]
        if len(records) < 10:
            return  # Not enough data

        expectancy = sum(r.pnl for r in records) / len(records)
        old_status = self._env_status.get(label, "ALLOWED")

        if expectancy < 0:
            new_status = "BLOCKED"
        else:
            new_status = "ALLOWED"

        self._env_status[label] = new_status

        if new_status != old_status:
            with open(self._log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.utcnow().isoformat(),
                    label, f"{expectancy:.2f}",
                    len(records), new_status,
                    f"status_change_{old_status}_to_{new_status}",
                ])

    def _check_week_rollover(self):
        """Reset weekly tracking on new week."""
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        if self._week_start != week_start:
            self._week_start = week_start
            self._symbol_weekly_losses.clear()
            self._symbol_weekly_trades.clear()
            self._disabled_symbols.clear()

    def disable_symbol_if_needed(self, symbol: str, max_weekly_loss: float = 500.0):
        """Disable a symbol if weekly losses exceed threshold."""
        if self._symbol_weekly_losses.get(symbol, 0.0) > max_weekly_loss:
            if symbol not in self._disabled_symbols:
                self._disabled_symbols.append(symbol)
                with open(self._log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        datetime.utcnow().isoformat(),
                        "SYMBOL_DISABLED", symbol,
                        self._symbol_weekly_losses[symbol],
                        "BLOCKED", "weekly_loss_exceeded",
                    ])
