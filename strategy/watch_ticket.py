"""
Watch Ticket System for V14.2 Core Runner

A watch ticket is created when a symbol-route candidate passes the watch admission gate.
Watch tickets NEVER submit orders. They track candidates through WATCHING → CONFIRMED → FILLED/EXPIRED/etc.
"""

from __future__ import annotations

import uuid
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG


class TicketStatus(Enum):
    WATCHING = "WATCHING"
    CONFIRMED = "CONFIRMED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    MISSED_FILL = "MISSED_FILL"
    CLOSED = "CLOSED"


class TicketSide(Enum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class WatchTicket:
    """
    Represents a watched candidate from base signal detection through lifecycle.
    A watch ticket must NOT submit orders.
    """
    ticket_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    route: str = ""
    side: TicketSide = TicketSide.CALL
    timestamp_created: Optional[datetime] = None
    underlying_price_at_watch: float = 0.0
    vwap_at_watch: float = 0.0
    atr_at_watch: float = 0.0
    route_score: float = 0.0
    ic_spread: float = 0.0
    expected_ev_over_debit: float = 0.0
    option_liquidity_score: float = 0.0
    expected_move_to_target: float = 0.0
    estimated_debit_at_watch: float = 0.0
    candidate_contracts: List[str] = field(default_factory=list)
    environment_state: Dict[str, Any] = field(default_factory=dict)
    status: TicketStatus = TicketStatus.WATCHING

    # Dynamic gate state (see strategy.dynamic_gate_controller). Populated
    # at ticket creation; drives position sizing and probation bookkeeping
    # downstream in execution.
    route_gate_state: str = "ACTIVE"       # ACTIVE, SOFT_SIZE_ONLY, PROBATION
    env_gate_state: str = "ALLOWED"        # ALLOWED, PROBATION
    size_multiplier: float = 1.0           # applied to package/fallback sizing

    # Confirmation data (populated on confirm)
    timestamp_confirmed: Optional[datetime] = None
    directional_move_atr: float = 0.0
    adverse_move_atr: float = 0.0
    mfe_velocity: float = 0.0
    confirmation_price: float = 0.0

    # Outcome data
    timestamp_closed: Optional[datetime] = None
    exit_reason: str = ""
    realized_pnl: float = 0.0

    def passes_watch_admission(self, cfg_override: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if this candidate passes watch-stage admission gates.

        cfg_override lets the caller supply per-route, confidence-adjusted
        thresholds (see strategy.dynamic_gate_controller) instead of the
        static base config; any key not present falls back to the base CFG.
        """
        cfg = cfg_override or {}
        min_route_score = cfg.get("watch_min_route_score", CFG["watch_min_route_score"])
        min_ic_spread = cfg.get("watch_min_ic_spread", CFG["watch_min_ic_spread"])
        min_ev_over_debit = cfg.get("watch_min_ev_over_debit", CFG["watch_min_ev_over_debit"])
        min_option_liquidity = cfg.get("watch_min_option_liquidity", CFG["watch_min_option_liquidity"])

        if self.route_score < min_route_score:
            return False
        if self.ic_spread < min_ic_spread:
            return False
        if self.expected_ev_over_debit < min_ev_over_debit:
            return False
        if self.option_liquidity_score < min_option_liquidity:
            return False
        return True

    def is_route_allowed(self) -> bool:
        """Check if route is allowed for full package or soft-only."""
        return self.route in CFG["routes_allowed"] or self.route in CFG["routes_soft_only"]

    def is_package_eligible_route(self) -> bool:
        """Only routes_allowed qualify for full core-runner package."""
        return self.route in CFG["routes_allowed"]

    def mark_confirmed(self, directional_move_atr: float, adverse_move_atr: float,
                       mfe_velocity: float, price: float, timestamp: Optional[datetime] = None):
        """Transition ticket to CONFIRMED state."""
        self.status = TicketStatus.CONFIRMED
        self.timestamp_confirmed = timestamp or datetime.utcnow()
        self.directional_move_atr = directional_move_atr
        self.adverse_move_atr = adverse_move_atr
        self.mfe_velocity = mfe_velocity
        self.confirmation_price = price

    def mark_expired(self, reason: str = "timeout"):
        self.status = TicketStatus.EXPIRED
        self.exit_reason = reason
        self.timestamp_closed = datetime.utcnow()

    def mark_rejected(self, reason: str):
        self.status = TicketStatus.REJECTED
        self.exit_reason = reason
        self.timestamp_closed = datetime.utcnow()

    def mark_filled(self):
        self.status = TicketStatus.FILLED

    def mark_missed_fill(self, reason: str):
        self.status = TicketStatus.MISSED_FILL
        self.exit_reason = reason
        self.timestamp_closed = datetime.utcnow()

    def mark_closed(self, realized_pnl: float, reason: str):
        self.status = TicketStatus.CLOSED
        self.realized_pnl = realized_pnl
        self.exit_reason = reason
        self.timestamp_closed = datetime.utcnow()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["status"] = self.status.value
        if self.timestamp_created:
            d["timestamp_created"] = self.timestamp_created.isoformat()
        if self.timestamp_confirmed:
            d["timestamp_confirmed"] = self.timestamp_confirmed.isoformat()
        if self.timestamp_closed:
            d["timestamp_closed"] = self.timestamp_closed.isoformat()
        return d


class WatchTicketBook:
    """
    Manages the lifecycle of all active and historical watch tickets.
    Provides logging of every ticket state transition.
    """

    def __init__(self, log_dir: str = "HFT/logs/v14_2"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active_tickets: Dict[str, WatchTicket] = {}
        self.history: List[WatchTicket] = []
        self._log_path = self.log_dir / "watch_tickets.csv"
        self._init_log()

    def _init_log(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "ticket_id", "symbol", "route", "side",
                    "status", "route_score", "ev_over_debit",
                    "underlying_price", "vwap", "atr",
                    "directional_move_atr", "adverse_move_atr", "mfe_velocity",
                    "exit_reason", "realized_pnl"
                ])

    def create_ticket(
        self, cfg_override: Optional[Dict[str, float]] = None, **kwargs
    ) -> Optional[WatchTicket]:
        """
        Attempt to create a watch ticket. Returns ticket if admission passes, else None.
        Base signal creates watch ticket ONLY. Watch ticket does NOT submit order.

        cfg_override: optional per-route confidence-adjusted admission
        thresholds from DynamicGateController; falls back to base config.
        """
        ticket = WatchTicket(**kwargs)
        if ticket.timestamp_created is None:
            ticket.timestamp_created = datetime.utcnow()

        if not ticket.passes_watch_admission(cfg_override):
            self._log_event(ticket, "REJECTED_AT_WATCH")
            return None

        if not ticket.is_route_allowed():
            self._log_event(ticket, "REJECTED_ROUTE_NOT_ALLOWED")
            return None

        self.active_tickets[ticket.ticket_id] = ticket
        self._log_event(ticket, "WATCHING")
        return ticket

    def get_watching_tickets(self, symbol: Optional[str] = None) -> List[WatchTicket]:
        """Get all tickets currently in WATCHING state."""
        tickets = [t for t in self.active_tickets.values() if t.status == TicketStatus.WATCHING]
        if symbol:
            tickets = [t for t in tickets if t.symbol == symbol]
        return tickets

    def get_confirmed_tickets(self, symbol: Optional[str] = None) -> List[WatchTicket]:
        """Get all tickets currently in CONFIRMED state."""
        tickets = [t for t in self.active_tickets.values() if t.status == TicketStatus.CONFIRMED]
        if symbol:
            tickets = [t for t in tickets if t.symbol == symbol]
        return tickets

    def retire_ticket(self, ticket_id: str):
        """Move ticket from active to history."""
        if ticket_id in self.active_tickets:
            ticket = self.active_tickets.pop(ticket_id)
            self.history.append(ticket)
            self._log_event(ticket, ticket.status.value)

    def expire_stale_tickets(self, max_watch_bars: int = 30):
        """Expire tickets that have been watching too long."""
        now = datetime.utcnow()
        for tid, ticket in list(self.active_tickets.items()):
            if ticket.status != TicketStatus.WATCHING:
                continue
            if ticket.timestamp_created:
                age_seconds = (now - ticket.timestamp_created).total_seconds()
                if age_seconds > max_watch_bars * 60:
                    ticket.mark_expired("watch_timeout")
                    self.retire_ticket(tid)

    def _log_event(self, ticket: WatchTicket, event: str):
        with open(self._log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.utcnow().isoformat(),
                ticket.ticket_id,
                ticket.symbol,
                ticket.route,
                ticket.side.value,
                event,
                f"{ticket.route_score:.4f}",
                f"{ticket.expected_ev_over_debit:.4f}",
                f"{ticket.underlying_price_at_watch:.2f}",
                f"{ticket.vwap_at_watch:.2f}",
                f"{ticket.atr_at_watch:.4f}",
                f"{ticket.directional_move_atr:.4f}",
                f"{ticket.adverse_move_atr:.4f}",
                f"{ticket.mfe_velocity:.4f}",
                ticket.exit_reason,
                f"{ticket.realized_pnl:.2f}",
            ])
