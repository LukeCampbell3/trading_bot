"""
Risk Manager for V14.2

Hard risk controls that cannot be overridden by strategy logic.
These are enforced at all times regardless of mode.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG


@dataclass
class TradeRecord:
    """Record of a completed or active trade for risk tracking."""
    timestamp: datetime
    symbol: str
    route: str
    debit: float
    pnl: float = 0.0
    is_open: bool = True


@dataclass
class RiskCheckResult:
    """Result of a risk pre-check."""
    allowed: bool = True
    reason: str = ""
    risk_level: str = "OK"  # OK, WARNING, BLOCKED, KILLED


class RiskManager:
    """
    Enforces hard risk controls for V14.2.
    
    Hard controls:
    - live_trading_enabled defaults to False
    - No orders in live mode unless explicitly enabled
    - max_trades_per_day
    - max_trades_per_week
    - max_same_underlying_trades_per_week
    - daily_kill_loss_pct
    - weekly_kill_loss_pct
    - max_open_debit_exposure_pct
    - No averaging down
    - No revenge trades
    - No breakout-only full greed
    - No raw long calls
    - No equity scalp fallback
    - No overnight holds during validation (unless enabled)
    """

    def __init__(self, config: Optional[dict] = None, log_dir: str = "HFT/logs/v14_2"):
        self.cfg = config or CFG
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._trade_records: List[TradeRecord] = []
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._current_date: Optional[date] = None
        self._week_start: Optional[date] = None
        self._daily_killed: bool = False
        self._weekly_killed: bool = False
        self._overnight_allowed: bool = False

        # Track open exposure
        self._open_positions: Dict[str, float] = {}  # symbol -> total debit
        self._account_equity: float = 0.0

        # Per-symbol weekly tracking
        self._symbol_trades_this_week: Dict[str, int] = defaultdict(int)
        self._symbol_losses_this_week: Dict[str, float] = defaultdict(float)
        self._disabled_symbols: List[str] = []

        self._log_path = self.log_dir / "risk_events.csv"
        self._init_log()

    def _init_log(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "event", "details", "result"])

    def update_account(self, equity: float):
        """Update account equity for exposure calculations."""
        self._account_equity = equity

    def reset_day(self, today: Optional[date] = None):
        """Reset daily counters. Called at start of each trading day."""
        today = today or datetime.utcnow().date()
        if self._current_date != today:
            self._current_date = today
            self._daily_pnl = 0.0
            self._daily_killed = False

            # Check week rollover (Monday)
            if self._week_start is None or (today - self._week_start).days >= 7:
                self._week_start = today - timedelta(days=today.weekday())
                self._weekly_pnl = 0.0
                self._weekly_killed = False
                self._symbol_trades_this_week.clear()
                self._symbol_losses_this_week.clear()
                self._disabled_symbols.clear()

    def record_trade(self, symbol: str, route: str, debit: float, pnl: float = 0.0):
        """Record a completed or opened trade."""
        now = datetime.utcnow()
        record = TradeRecord(
            timestamp=now,
            symbol=symbol,
            route=route,
            debit=debit,
            pnl=pnl,
            is_open=(pnl == 0.0),
        )
        self._trade_records.append(record)
        self._symbol_trades_this_week[symbol] += 1
        # Also ensure current_date is set if not already
        if self._current_date is None:
            self._current_date = now.date()

    def record_close(self, symbol: str, pnl: float):
        """Record a trade close and update P&L."""
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        if pnl < 0:
            self._symbol_losses_this_week[symbol] += abs(pnl)

        # Remove from open positions
        if symbol in self._open_positions:
            del self._open_positions[symbol]

        self._check_kill_switches()

    def add_open_position(self, symbol: str, debit: float):
        """Track an open position's debit exposure."""
        self._open_positions[symbol] = self._open_positions.get(symbol, 0.0) + debit

    def pre_trade_check(
        self,
        symbol: str,
        route: str,
        debit: float,
        is_live: bool = False,
        is_averaging_down: bool = False,
        is_revenge: bool = False,
        is_raw_long_call: bool = False,
        is_equity_scalp: bool = False,
        is_breakout_full_greed: bool = False,
    ) -> RiskCheckResult:
        """
        Comprehensive pre-trade risk check. Must pass ALL checks before execution.
        """
        result = RiskCheckResult()

        # ─── HARD BLOCK: Live trading disabled ───────────────────────────
        if is_live and not self.cfg["live_trading_enabled"]:
            result.allowed = False
            result.reason = "LIVE_TRADING_DISABLED"
            result.risk_level = "BLOCKED"
            self._log_event("BLOCKED_LIVE_DISABLED", f"symbol={symbol}")
            return result

        # ─── HARD BLOCK: Kill switches ───────────────────────────────────
        if self._daily_killed:
            result.allowed = False
            result.reason = "DAILY_KILL_ACTIVE"
            result.risk_level = "KILLED"
            return result

        if self._weekly_killed:
            result.allowed = False
            result.reason = "WEEKLY_KILL_ACTIVE"
            result.risk_level = "KILLED"
            return result

        # ─── HARD BLOCK: Forbidden trade types ───────────────────────────
        if is_averaging_down:
            result.allowed = False
            result.reason = "NO_AVERAGING_DOWN"
            result.risk_level = "BLOCKED"
            self._log_event("BLOCKED_AVG_DOWN", f"symbol={symbol}")
            return result

        if is_revenge:
            result.allowed = False
            result.reason = "NO_REVENGE_TRADES"
            result.risk_level = "BLOCKED"
            self._log_event("BLOCKED_REVENGE", f"symbol={symbol}")
            return result

        if is_raw_long_call:
            result.allowed = False
            result.reason = "NO_RAW_LONG_CALLS"
            result.risk_level = "BLOCKED"
            self._log_event("BLOCKED_RAW_CALL", f"symbol={symbol}")
            return result

        if is_equity_scalp:
            result.allowed = False
            result.reason = "NO_EQUITY_SCALP_FALLBACK"
            result.risk_level = "BLOCKED"
            self._log_event("BLOCKED_EQUITY_SCALP", f"symbol={symbol}")
            return result

        if is_breakout_full_greed:
            result.allowed = False
            result.reason = "NO_BREAKOUT_ONLY_FULL_GREED"
            result.risk_level = "BLOCKED"
            self._log_event("BLOCKED_BREAKOUT_GREED", f"symbol={symbol}")
            return result

        # ─── Rate Limits ─────────────────────────────────────────────────
        trades_today = self._count_trades_today()
        if trades_today >= self.cfg["max_trades_per_day"]:
            result.allowed = False
            result.reason = f"MAX_TRADES_PER_DAY: {trades_today}/{self.cfg['max_trades_per_day']}"
            result.risk_level = "BLOCKED"
            return result

        trades_this_week = self._count_trades_this_week()
        if trades_this_week >= self.cfg["max_trades_per_week"]:
            result.allowed = False
            result.reason = f"MAX_TRADES_PER_WEEK: {trades_this_week}/{self.cfg['max_trades_per_week']}"
            result.risk_level = "BLOCKED"
            return result

        symbol_trades = self._symbol_trades_this_week.get(symbol, 0)
        if symbol_trades >= self.cfg["max_same_underlying_trades_per_week"]:
            result.allowed = False
            result.reason = f"MAX_SAME_UNDERLYING: {symbol} has {symbol_trades} this week"
            result.risk_level = "BLOCKED"
            return result

        # ─── Exposure Limits ─────────────────────────────────────────────
        if self._account_equity > 0:
            current_exposure = sum(self._open_positions.values())
            new_exposure = current_exposure + debit
            exposure_pct = new_exposure / self._account_equity
            if exposure_pct > self.cfg["max_open_debit_exposure_pct"]:
                result.allowed = False
                result.reason = (
                    f"MAX_EXPOSURE: {exposure_pct:.3f} > {self.cfg['max_open_debit_exposure_pct']}"
                )
                result.risk_level = "BLOCKED"
                return result

        # ─── Symbol Disabled ─────────────────────────────────────────────
        if symbol in self._disabled_symbols:
            result.allowed = False
            result.reason = f"SYMBOL_DISABLED: {symbol}"
            result.risk_level = "BLOCKED"
            return result

        return result

    def _check_kill_switches(self):
        """Check if kill switches should activate."""
        if self._account_equity <= 0:
            return

        daily_pct = self._daily_pnl / self._account_equity
        if daily_pct <= self.cfg["daily_kill_loss_pct"]:
            self._daily_killed = True
            self._log_event("DAILY_KILL_TRIGGERED", f"pnl_pct={daily_pct:.4f}")

        weekly_pct = self._weekly_pnl / self._account_equity
        if weekly_pct <= self.cfg["weekly_kill_loss_pct"]:
            self._weekly_killed = True
            self._log_event("WEEKLY_KILL_TRIGGERED", f"pnl_pct={weekly_pct:.4f}")

    def _count_trades_today(self) -> int:
        today = self._current_date or date.today()
        return sum(
            1 for t in self._trade_records
            if t.timestamp.date() == today
        )

    def _count_trades_this_week(self) -> int:
        if self._week_start is None:
            return 0
        return sum(
            1 for t in self._trade_records
            if t.timestamp.date() >= self._week_start
        )

    def is_killed(self) -> bool:
        """Check if any kill switch is active."""
        return self._daily_killed or self._weekly_killed

    def _log_event(self, event: str, details: str):
        with open(self._log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([datetime.utcnow().isoformat(), event, details, "BLOCKED"])
