"""
Slippage Monitor for V14.2

Monitors fill slippage and missed fill rates.
Implements adaptive rules:
- If missed_fill_rate exceeds modeled rate by 50%, reduce limit aggressiveness and size
- If exit slippage exceeds modeled stress by 50%, reduce targets and runner exposure
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict


@dataclass
class SlippageConfig:
    """Adaptive slippage parameters."""
    # Entry adjustments
    limit_aggressiveness_factor: float = 1.0  # 1.0 = normal, <1.0 = more conservative
    size_reduction_factor: float = 1.0  # 1.0 = normal, <1.0 = reduced size

    # Exit adjustments
    target_reduction_factor: float = 1.0  # 1.0 = normal, <1.0 = lower targets
    runner_exposure_factor: float = 1.0  # 1.0 = normal, <1.0 = less runner

    @property
    def is_degraded(self) -> bool:
        return (
            self.limit_aggressiveness_factor < 1.0
            or self.size_reduction_factor < 1.0
            or self.target_reduction_factor < 1.0
            or self.runner_exposure_factor < 1.0
        )


class SlippageMonitor:
    """
    Monitors entry/exit slippage and adapts execution parameters.
    """

    def __init__(
        self,
        modeled_missed_fill_rate: float = 0.15,
        modeled_exit_slippage: float = 0.02,
        log_dir: str = "HFT/logs/v14_2",
        window: int = 50,
    ):
        self.modeled_missed_fill_rate = modeled_missed_fill_rate
        self.modeled_exit_slippage = modeled_exit_slippage
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._window = window

        # Rolling data
        self._entry_slippages: deque = deque(maxlen=window)
        self._exit_slippages: deque = deque(maxlen=window)
        self._fill_attempts: deque = deque(maxlen=window)  # True=filled, False=missed
        self._mid_inflation_rates: deque = deque(maxlen=window)

        # Current adaptive config
        self.config = SlippageConfig()

        self._log_path = self.log_dir / "slippage_monitor.csv"
        self._init_log()

    def _init_log(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "event",
                    "entry_slip_mean", "exit_slip_mean",
                    "missed_fill_rate", "mid_inflation_mean",
                    "limit_factor", "size_factor",
                    "target_factor", "runner_factor",
                ])

    def record_entry_fill(self, slippage_vs_mid: float, was_filled: bool):
        """Record an entry attempt and its fill quality."""
        self._fill_attempts.append(was_filled)
        if was_filled:
            self._entry_slippages.append(slippage_vs_mid)
        self._evaluate_entry_adaptive()

    def record_exit_fill(self, slippage_vs_bid: float):
        """Record an exit fill and its slippage."""
        self._exit_slippages.append(slippage_vs_bid)
        self._evaluate_exit_adaptive()

    def record_mid_inflation(self, inflation_rate: float):
        """Record option mid inflation rate."""
        self._mid_inflation_rates.append(inflation_rate)

    @property
    def current_missed_fill_rate(self) -> float:
        if not self._fill_attempts:
            return 0.0
        return sum(1 for f in self._fill_attempts if not f) / len(self._fill_attempts)

    @property
    def current_entry_slippage(self) -> float:
        if not self._entry_slippages:
            return 0.0
        return sum(self._entry_slippages) / len(self._entry_slippages)

    @property
    def current_exit_slippage(self) -> float:
        if not self._exit_slippages:
            return 0.0
        return sum(self._exit_slippages) / len(self._exit_slippages)

    @property
    def current_mid_inflation(self) -> float:
        if not self._mid_inflation_rates:
            return 0.0
        return sum(self._mid_inflation_rates) / len(self._mid_inflation_rates)

    def _evaluate_entry_adaptive(self):
        """Adapt entry parameters if missed fill rate exceeds model."""
        if len(self._fill_attempts) < 10:
            return

        actual_rate = self.current_missed_fill_rate
        threshold = self.modeled_missed_fill_rate * 1.5

        if actual_rate > threshold:
            # Reduce aggressiveness: widen limit tolerance
            degradation = min(0.3, (actual_rate - threshold) / threshold)
            self.config.limit_aggressiveness_factor = max(0.7, 1.0 - degradation)
            self.config.size_reduction_factor = max(0.7, 1.0 - degradation * 0.5)
            self._log_adaptation("ENTRY_DEGRADED")
        else:
            # Slowly recover
            self.config.limit_aggressiveness_factor = min(
                1.0, self.config.limit_aggressiveness_factor + 0.02
            )
            self.config.size_reduction_factor = min(
                1.0, self.config.size_reduction_factor + 0.02
            )

    def _evaluate_exit_adaptive(self):
        """Adapt exit parameters if slippage exceeds model."""
        if len(self._exit_slippages) < 10:
            return

        actual_slip = self.current_exit_slippage
        threshold = self.modeled_exit_slippage * 1.5

        if actual_slip > threshold:
            degradation = min(0.3, (actual_slip - threshold) / threshold)
            self.config.target_reduction_factor = max(0.7, 1.0 - degradation)
            self.config.runner_exposure_factor = max(0.5, 1.0 - degradation * 1.5)
            self._log_adaptation("EXIT_DEGRADED")
        else:
            self.config.target_reduction_factor = min(
                1.0, self.config.target_reduction_factor + 0.02
            )
            self.config.runner_exposure_factor = min(
                1.0, self.config.runner_exposure_factor + 0.02
            )

    def _log_adaptation(self, event: str):
        with open(self._log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.utcnow().isoformat(), event,
                f"{self.current_entry_slippage:.4f}",
                f"{self.current_exit_slippage:.4f}",
                f"{self.current_missed_fill_rate:.4f}",
                f"{self.current_mid_inflation:.4f}",
                f"{self.config.limit_aggressiveness_factor:.3f}",
                f"{self.config.size_reduction_factor:.3f}",
                f"{self.config.target_reduction_factor:.3f}",
                f"{self.config.runner_exposure_factor:.3f}",
            ])

    def get_summary(self) -> dict:
        return {
            "entry_slippage_mean": self.current_entry_slippage,
            "exit_slippage_mean": self.current_exit_slippage,
            "missed_fill_rate": self.current_missed_fill_rate,
            "mid_inflation_mean": self.current_mid_inflation,
            "is_degraded": self.config.is_degraded,
            "limit_factor": self.config.limit_aggressiveness_factor,
            "size_factor": self.config.size_reduction_factor,
            "target_factor": self.config.target_reduction_factor,
            "runner_factor": self.config.runner_exposure_factor,
        }
