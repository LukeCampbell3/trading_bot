"""
execution_controller.py  --  V14_2_1 Loss-Governed Active Portfolio Controller
=================================================================================
Signals do not trade. Signals request exposure.
The execution controller owns orders. Broker state is the source of truth.

Symbol States:
    FLAT -> WATCH -> BUILDING -> ACTIVE -> PROTECT_PROFIT -> REDUCING -> EXITING -> COOLDOWN_BLOCKED
    ACTIVE -> REVALIDATE_LOSER -> REDUCING -> EXITING
    Any -> LOCKED_ERROR

Loss Governance Thresholds:
    -1.50%  -> REVALIDATE_LOSER (revalidate signal)
    -2.50%  -> fresh confirm required or exit
    -4.00%  -> target zero immediately
    -6.00%  -> lock session (LOCKED_ERROR)

Profit Protection:
    +0.75%  -> no adding (hold only)
    +1.25%  -> PROTECT_PROFIT state
    +2.00%  -> trim/trail

Controller Loop Order:
    broker truth -> rebuild ledgers -> detect contradictions -> cancel stale ->
    refresh -> rescore positions -> recompute targets -> reduce/exit -> refresh ->
    block if unresolved -> generate candidates -> admit -> assign targets ->
    submit buys -> audit

Client Order ID Format:
    BOT|V14_2_1|SYMBOL|CAMPAIGN_ID|SIDE|TIMESTAMP|UUID
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================================
# Constants & Enums
# ============================================================================

VERSION = "V14_2_1"


class SymbolState(str, Enum):
    FLAT = "FLAT"
    WATCH = "WATCH"
    BUILDING = "BUILDING"
    ACTIVE = "ACTIVE"
    PROTECT_PROFIT = "PROTECT_PROFIT"
    REVALIDATE_LOSER = "REVALIDATE_LOSER"
    REDUCING = "REDUCING"
    EXITING = "EXITING"
    COOLDOWN_BLOCKED = "COOLDOWN_BLOCKED"
    LOCKED_ERROR = "LOCKED_ERROR"


class SafetyMode(str, Enum):
    NORMAL = "NORMAL"
    LOSS_MITIGATION = "LOSS_MITIGATION"
    ORDER_RECONCILIATION_ONLY = "ORDER_RECONCILIATION_ONLY"
    NO_NEW_ENTRIES = "NO_NEW_ENTRIES"
    LIQUIDATE_ONLY = "LIQUIDATE_ONLY"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


# Forbidden transitions: (from_state, to_state) pairs that must NEVER occur
FORBIDDEN_TRANSITIONS: Set[Tuple[str, str]] = {
    ("EXITING", "BUILDING"),
    ("REDUCING", "BUILDING"),
    ("LOCKED_ERROR", "BUILDING"),
    ("COOLDOWN_BLOCKED", "BUILDING"),
    ("REVALIDATE_LOSER", "BUILDING"),   # Cannot add size to a loser
    ("LOCKED_ERROR", "ACTIVE"),
    ("EXITING", "ACTIVE"),
    ("COOLDOWN_BLOCKED", "ACTIVE"),
    ("LOCKED_ERROR", "WATCH"),
}


# Loss governance thresholds (as fractions, negative = loss)
LOSS_REVALIDATE_PCT = -0.0150       # -1.50%
LOSS_FRESH_CONFIRM_PCT = -0.0250    # -2.50%
LOSS_TARGET_ZERO_PCT = -0.0400      # -4.00%
LOSS_LOCK_SESSION_PCT = -0.0600     # -6.00%

# Profit protection thresholds
PROFIT_NO_ADDING_PCT = 0.0075       # +0.75%
PROFIT_PROTECT_PCT = 0.0125         # +1.25%
PROFIT_TRIM_TRAIL_PCT = 0.0200      # +2.00%

# Portfolio admission limits
MAX_ACTIVE_POSITIONS = 8
MAX_NEW_PER_CYCLE = 3
MAX_NEW_PER_DAY = 6
REPLACEMENT_MARGIN = 1.15           # new candidate must score 1.15x vs worst held


# ============================================================================
# Core Data Objects
# ============================================================================

@dataclass
class TradeSignal:
    """What the signal engine outputs. Signals REQUEST exposure -- they do not trade."""
    symbol: str
    bucket_id: str
    direction: str              # "long", "flat"
    score: float
    expected_return_bps: float
    setup_fingerprint: str
    generated_at: datetime
    ttl_seconds: int
    invalidation_price: Optional[float] = None
    reason: str = ""
    category: str = "general"   # for category-cap admission

    @property
    def is_expired(self) -> bool:
        age = (datetime.utcnow() - self.generated_at).total_seconds()
        return age > self.ttl_seconds


@dataclass
class SymbolCampaign:
    """Authoritative state per symbol. Tracks what exposure SHOULD be."""
    symbol: str
    state: str = "FLAT"
    campaign_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    target_qty: float = 0.0
    max_qty: float = 0.0

    bucket_id: Optional[str] = None
    setup_fingerprint: Optional[str] = None
    category: str = "general"

    last_exit_time: Optional[datetime] = None
    last_exit_reason: Optional[str] = None
    blocked_setup_fingerprint: Optional[str] = None

    invalidation_price: Optional[float] = None
    locked_reason: Optional[str] = None

    # Tracking
    entry_price: Optional[float] = None
    bars_held: int = 0
    best_pnl_pct: float = 0.0
    current_pnl_pct: float = 0.0
    last_signal_time: Optional[datetime] = None
    confirmations: int = 0
    cooldown_until: Optional[datetime] = None

    # Hold scorer inputs
    hold_score: float = 0.0
    trend_aligned: bool = False
    volume_confirmed: bool = False

    def transition_to(self, new_state: str) -> bool:
        """Attempt state transition. Returns False if forbidden."""
        if (self.state, new_state) in FORBIDDEN_TRANSITIONS:
            return False
        self.state = new_state
        return True

    def reset_flat(self):
        """Full reset to FLAT state."""
        self.state = "FLAT"
        self.target_qty = 0.0
        self.max_qty = 0.0
        self.bucket_id = None
        self.setup_fingerprint = None
        self.invalidation_price = None
        self.locked_reason = None
        self.entry_price = None
        self.bars_held = 0
        self.best_pnl_pct = 0.0
        self.current_pnl_pct = 0.0
        self.confirmations = 0
        self.hold_score = 0.0
        self.campaign_id = uuid.uuid4().hex[:8]


@dataclass
class BrokerSymbolState:
    """Rebuilt from Alpaca every loop. This is broker truth."""
    symbol: str
    position_qty: int
    avg_entry_price: Optional[float]

    pending_buy_qty: int
    pending_sell_qty: int

    open_buy_order_ids: List[str] = field(default_factory=list)
    open_sell_order_ids: List[str] = field(default_factory=list)

    open_buy_orders: List[dict] = field(default_factory=list)
    open_sell_orders: List[dict] = field(default_factory=list)

    @property
    def effective_exposure(self) -> int:
        return self.position_qty + self.pending_buy_qty - self.pending_sell_qty


@dataclass
class BrokerSnapshot:
    """Full broker truth at a point in time."""
    account_equity: float
    account_buying_power: float
    positions: Dict[str, BrokerSymbolState] = field(default_factory=dict)
    all_open_orders: List[dict] = field(default_factory=list)

    def get_symbol_state(self, symbol: str) -> BrokerSymbolState:
        if symbol in self.positions:
            return self.positions[symbol]
        return BrokerSymbolState(
            symbol=symbol, position_qty=0, avg_entry_price=None,
            pending_buy_qty=0, pending_sell_qty=0)

    @property
    def total_exposure_notional(self) -> float:
        total = 0.0
        for bs in self.positions.values():
            if bs.avg_entry_price and bs.position_qty > 0:
                total += bs.position_qty * bs.avg_entry_price
        return total

    @property
    def n_active_positions(self) -> int:
        return sum(1 for bs in self.positions.values() if bs.position_qty > 0)


@dataclass
class ExecutionPolicy:
    """All tunable knobs for the execution controller."""
    # Session
    allow_extended_hours: bool = False
    no_new_entries_before: str = "09:35"
    cancel_premarket_orders_at: str = "09:29"
    no_new_entries_after: str = "15:55"

    # Sizing
    max_risk_per_trade_pct: float = 0.0035
    max_symbol_notional_pct: float = 0.18
    max_shares_per_symbol: int = 5000
    atr_stop_multiple: float = 2.2
    default_stop_pct: float = 0.005
    take_profit_multiple: float = 1.5

    # Portfolio
    max_total_exposure_pct: float = 0.85
    max_concurrent_positions: int = MAX_ACTIVE_POSITIONS
    max_trades_per_day: int = 80
    daily_max_drawdown_pct: float = 0.035
    max_new_per_cycle: int = MAX_NEW_PER_CYCLE
    max_new_per_day: int = MAX_NEW_PER_DAY
    replacement_margin: float = REPLACEMENT_MARGIN

    # Category caps (category -> max positions)
    category_caps: Dict[str, int] = field(default_factory=lambda: {
        "tech": 3, "meme": 2, "crypto_related": 2, "general": 4,
    })

    # No-trade-zone
    min_net_edge_bps: float = 3.0

    # Order management
    default_order_ttl_seconds: int = 90
    order_ttl_seconds_by_bucket: Dict[str, int] = field(default_factory=lambda: {
        "premarket": 30, "open_drive": 20, "momentum": 60,
        "pullback": 120, "default": 90,
    })

    # Re-entry
    same_setup_reentry_score: float = 0.80
    loss_reentry_score: float = 0.75
    structure_reset_bars: int = 15
    cooldown_seconds: int = 300

    # Bucket -> target scale
    bucket_target_scale: Dict[str, float] = field(default_factory=lambda: {
        "B0": 0.00, "B1": 0.00, "B2": 0.00,
        "B3": 0.35, "B4": 0.70, "B5": 1.00,
    })


@dataclass
class CostEstimate:
    """Per-signal cost breakdown."""
    spread_bps: float = 3.0
    slippage_bps: float = 1.5
    churn_penalty_bps: float = 0.0
    stale_penalty_bps: float = 0.0


@dataclass
class MarketState:
    """Per-symbol market structure context for re-entry qualification."""
    bars_since_last_exit: int = 0
    structure_reset: bool = False
    continuation_confirmed: bool = False
    atr: float = 0.0
    last_price: float = 0.0


@dataclass
class Decision:
    """Output of the decision engine."""
    action: str             # "SET_TARGET", "TARGET_ZERO", "LOCK", "NO_ACTION"
    target_qty: float = 0.0
    reason: str = ""


# ============================================================================
# Setup Fingerprinting & Order ID
# ============================================================================

def compute_setup_fingerprint(symbol: str, bucket_id: str, direction: str,
                              z_score: float, vol_regime: str, trend_state: str) -> str:
    """Fingerprint for a specific setup structure. Same fingerprint = same trade thesis."""
    raw = f"{symbol}|{bucket_id}|{direction}|z{z_score:.1f}|{vol_regime}|{trend_state}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def generate_client_order_id(symbol: str, campaign_id: str, side: str,
                             setup_fingerprint: str = "") -> str:
    """
    Client order ID format: BOT|V14_2_1|SYMBOL|CAMPAIGN_ID|SIDE|TIMESTAMP|UUID
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    return f"BOT|V14_2_1|{symbol}|{campaign_id}|{side}|{ts}|{short_id}"


def is_bot_order(client_order_id: str) -> bool:
    """Check if an order was placed by this bot (supports V15 and V14_2_1 prefixes)."""
    if not client_order_id:
        return False
    return client_order_id.startswith("BOT|V15|") or client_order_id.startswith("BOT|V14_2_1|")


# ============================================================================
# No-Trade-Zone Gate
# ============================================================================

def net_edge_bps(signal: TradeSignal, costs: CostEstimate) -> float:
    return (
        signal.expected_return_bps
        - costs.spread_bps
        - costs.slippage_bps
        - costs.churn_penalty_bps
        - costs.stale_penalty_bps
    )


def passes_no_trade_zone(signal: TradeSignal, costs: CostEstimate,
                         policy: ExecutionPolicy) -> Tuple[bool, str]:
    edge = net_edge_bps(signal, costs)
    if edge < policy.min_net_edge_bps:
        return False, f"net_edge_too_small({edge:.1f}bps < {policy.min_net_edge_bps}bps)"
    return True, f"edge_accepted({edge:.1f}bps)"


# ============================================================================
# Re-Entry Qualification
# ============================================================================

def passes_reentry_gate(signal: TradeSignal, campaign: SymbolCampaign,
                        market_state: MarketState, policy: ExecutionPolicy) -> Tuple[bool, str]:
    """Replaces simple cooldown with structural re-entry check."""
    # Cooldown check
    if campaign.cooldown_until and datetime.utcnow() < campaign.cooldown_until:
        return False, "cooldown_active"

    if campaign.blocked_setup_fingerprint is None:
        return True, "no_blocked_setup"

    same_setup = signal.setup_fingerprint == campaign.blocked_setup_fingerprint

    if same_setup and not market_state.structure_reset:
        return False, "same_setup_no_reset"

    if same_setup and signal.score < policy.same_setup_reentry_score:
        return False, f"same_setup_score_weak({signal.score:.2f}<{policy.same_setup_reentry_score})"

    if campaign.last_exit_reason == "loss" and signal.score < policy.loss_reentry_score:
        return False, f"post_loss_score_weak({signal.score:.2f})"

    if campaign.last_exit_reason == "profit" and not market_state.continuation_confirmed:
        return False, "profit_exit_no_continuation"

    return True, "reentry_approved"


# ============================================================================
# Target Sizing
# ============================================================================

def compute_target_qty(signal: TradeSignal, account_equity: float, price: float,
                       atr: float, policy: ExecutionPolicy,
                       buying_power: float = None) -> float:
    """
    Risk-based target sizing. Returns fractional qty (supports small accounts).
    Bucket scale modulates conviction. Caps by actual buying_power.
    """
    scale = policy.bucket_target_scale.get(signal.bucket_id, 0.0)
    if scale <= 0:
        return 0.0
    if price <= 0:
        return 0.0

    risk_budget = account_equity * policy.max_risk_per_trade_pct * scale

    if signal.invalidation_price and signal.invalidation_price > 0:
        per_share_risk = abs(price - signal.invalidation_price)
    else:
        per_share_risk = max(atr * policy.atr_stop_multiple, price * policy.default_stop_pct)

    if per_share_risk <= 0:
        return 0.0

    risk_qty = risk_budget / per_share_risk
    notional_cap = account_equity * policy.max_symbol_notional_pct * scale
    notional_qty = notional_cap / price

    # Cap by actual buying power
    bp_qty = (buying_power * 0.90) / price if buying_power and buying_power > 0 else notional_qty

    raw = min(risk_qty, notional_qty, bp_qty, float(policy.max_shares_per_symbol))

    # Round to 2 decimal places for fractional shares (Alpaca minimum is 0.001)
    result = round(raw, 2)
    return result if result >= 0.01 else 0.0


# ============================================================================
# Loss Governor
# ============================================================================

class LossGovernor:
    """
    Monitors P/L per campaign and enforces loss governance thresholds.
    Thresholds:
        -1.50%  -> revalidate (REVALIDATE_LOSER)
        -2.50%  -> fresh confirm required or exit
        -4.00%  -> target zero immediately
        -6.00%  -> lock session (LOCKED_ERROR)
    """

    def __init__(self, policy: ExecutionPolicy):
        self.policy = policy
        self.session_locked_symbols: Set[str] = set()

    def evaluate(self, campaign: SymbolCampaign, current_price: float) -> Optional[str]:
        """
        Returns action needed: None, 'revalidate', 'fresh_confirm', 'target_zero', 'lock_session'
        """
        if campaign.state in ("FLAT", "WATCH", "COOLDOWN_BLOCKED", "LOCKED_ERROR"):
            return None
        if campaign.entry_price is None or campaign.entry_price <= 0:
            return None
        if current_price <= 0:
            return None

        pnl_pct = (current_price - campaign.entry_price) / campaign.entry_price
        campaign.current_pnl_pct = pnl_pct
        campaign.best_pnl_pct = max(campaign.best_pnl_pct, pnl_pct)

        if campaign.symbol in self.session_locked_symbols:
            return "lock_session"

        # Small epsilon for floating-point boundary comparisons
        _eps = 1e-9
        if pnl_pct <= LOSS_LOCK_SESSION_PCT + _eps:
            self.session_locked_symbols.add(campaign.symbol)
            return "lock_session"
        elif pnl_pct <= LOSS_TARGET_ZERO_PCT + _eps:
            return "target_zero"
        elif pnl_pct <= LOSS_FRESH_CONFIRM_PCT + _eps:
            return "fresh_confirm"
        elif pnl_pct <= LOSS_REVALIDATE_PCT + _eps:
            return "revalidate"

        return None


    def evaluate_profit_protection(self, campaign: SymbolCampaign,
                                   current_price: float) -> Optional[str]:
        """
        Profit protection thresholds:
            +0.75%  -> no adding
            +1.25%  -> PROTECT_PROFIT
            +2.00%  -> trim/trail
        Returns: None, 'no_adding', 'protect_profit', 'trim_trail'
        """
        if campaign.entry_price is None or campaign.entry_price <= 0:
            return None
        if current_price <= 0:
            return None

        pnl_pct = (current_price - campaign.entry_price) / campaign.entry_price

        if pnl_pct >= PROFIT_TRIM_TRAIL_PCT:
            return "trim_trail"
        elif pnl_pct >= PROFIT_PROTECT_PCT:
            return "protect_profit"
        elif pnl_pct >= PROFIT_NO_ADDING_PCT:
            return "no_adding"

        return None

    def apply_loss_action(self, campaign: SymbolCampaign, action: str,
                          has_fresh_confirm: bool = False) -> str:
        """Apply the loss governance action to campaign. Returns new state."""
        if action == "lock_session":
            campaign.state = "LOCKED_ERROR"
            campaign.locked_reason = "loss_exceeded_6.0pct"
            campaign.target_qty = 0
            return "LOCKED_ERROR"
        elif action == "target_zero":
            campaign.target_qty = 0
            campaign.state = "EXITING"
            return "EXITING"
        elif action == "fresh_confirm":
            if not has_fresh_confirm:
                campaign.target_qty = 0
                campaign.state = "EXITING"
                return "EXITING"
            return campaign.state
        elif action == "revalidate":
            if campaign.state != "REVALIDATE_LOSER":
                campaign.transition_to("REVALIDATE_LOSER")
            return campaign.state
        return campaign.state

    def apply_profit_action(self, campaign: SymbolCampaign, action: str) -> str:
        """Apply profit protection to campaign. Returns new state."""
        if action == "trim_trail":
            # Trim to half position
            campaign.target_qty = round(max(0.01, campaign.target_qty / 2.0), 2)
            if campaign.state != "REDUCING":
                campaign.transition_to("REDUCING")
            return campaign.state
        elif action == "protect_profit":
            campaign.transition_to("PROTECT_PROFIT")
            return campaign.state
        elif action == "no_adding":
            # Don't change state, just block additions externally
            return campaign.state
        return campaign.state


# ============================================================================
# Hold Scorer
# ============================================================================

class HoldScorer:
    """
    Scores existing positions to rank for replacement decisions.
    Components:
        - bucket_quality: B5=1.0, B4=0.7, B3=0.35, B2=0.0
        - confirmation: number of re-confirmations (capped at 3)
        - trend: aligned with broader trend (+0.15)
        - volume: volume confirms direction (+0.10)
        - pnl: current P/L contribution (scaled)
        - stale_penalty: -0.05 per 10 bars without signal refresh
        - loss_penalty: -0.20 if in REVALIDATE_LOSER
        - spread_correlation: penalize high-spread or correlated positions
    """

    BUCKET_QUALITY = {"B5": 1.0, "B4": 0.7, "B3": 0.35, "B2": 0.0, "B1": 0.0, "B0": 0.0}

    def score(self, campaign: SymbolCampaign, current_price: float = 0.0,
              spread_bps: float = 3.0, correlation_penalty: float = 0.0) -> float:
        """Compute hold score for a campaign."""
        s = 0.0

        # Bucket quality
        s += self.BUCKET_QUALITY.get(campaign.bucket_id or "B0", 0.0)

        # Confirmation bonus (capped at 3 * 0.10 = 0.30)
        s += min(campaign.confirmations, 3) * 0.10

        # Trend alignment
        if campaign.trend_aligned:
            s += 0.15

        # Volume confirmation
        if campaign.volume_confirmed:
            s += 0.10

        # P/L contribution (scaled: +1% = +0.20, -1% = -0.20)
        if campaign.entry_price and campaign.entry_price > 0 and current_price > 0:
            pnl_pct = (current_price - campaign.entry_price) / campaign.entry_price
            s += pnl_pct * 20.0  # scale so 1% = 0.20

        # Stale penalty: -0.05 per 10 bars without refresh
        stale_intervals = campaign.bars_held // 10
        s -= stale_intervals * 0.05

        # Loss penalty
        if campaign.state == "REVALIDATE_LOSER":
            s -= 0.20

        # Spread / correlation penalty
        s -= (spread_bps / 100.0) * 0.05
        s -= correlation_penalty

        campaign.hold_score = s
        return s


# ============================================================================
# Safety Mode Manager
# ============================================================================

class SafetyModeManager:
    """
    Manages the global safety mode of the portfolio controller.
    Modes:
        NORMAL                    - full operation
        LOSS_MITIGATION           - no new entries, reduce losers
        ORDER_RECONCILIATION_ONLY - only reconcile, no new orders
        NO_NEW_ENTRIES            - exits only
        LIQUIDATE_ONLY            - flatten everything
        MANUAL_REVIEW_REQUIRED    - halt all automated actions
    """

    def __init__(self):
        self.mode: SafetyMode = SafetyMode.NORMAL
        self.mode_reason: str = ""
        self.mode_since: datetime = datetime.utcnow()
        self.escalation_count: int = 0

    def set_mode(self, mode: SafetyMode, reason: str = ""):
        self.mode = mode
        self.mode_reason = reason
        self.mode_since = datetime.utcnow()
        self.escalation_count += 1

    def allows_new_entries(self) -> bool:
        return self.mode == SafetyMode.NORMAL

    def allows_any_orders(self) -> bool:
        return self.mode not in (SafetyMode.MANUAL_REVIEW_REQUIRED,)

    def allows_exits(self) -> bool:
        return self.mode not in (SafetyMode.MANUAL_REVIEW_REQUIRED,)

    def is_liquidate_only(self) -> bool:
        return self.mode == SafetyMode.LIQUIDATE_ONLY

    def evaluate_portfolio_health(self, snapshot: BrokerSnapshot,
                                  campaigns: Dict[str, SymbolCampaign],
                                  day_start_equity: float) -> SafetyMode:
        """Auto-detect if safety mode should change based on portfolio health."""
        if day_start_equity <= 0:
            return self.mode

        dd = (snapshot.account_equity / day_start_equity) - 1.0

        # Escalation ladder
        if dd <= -0.05:
            self.set_mode(SafetyMode.LIQUIDATE_ONLY, f"drawdown={dd:.2%}")
            return self.mode
        elif dd <= -0.03:
            self.set_mode(SafetyMode.NO_NEW_ENTRIES, f"drawdown={dd:.2%}")
            return self.mode
        elif dd <= -0.02:
            self.set_mode(SafetyMode.LOSS_MITIGATION, f"drawdown={dd:.2%}")
            return self.mode

        # Count unresolved contradictions / locked
        locked_count = sum(1 for c in campaigns.values() if c.state == "LOCKED_ERROR")
        if locked_count >= 3:
            self.set_mode(SafetyMode.ORDER_RECONCILIATION_ONLY, f"locked_count={locked_count}")
            return self.mode

        # If previously escalated and conditions improved, de-escalate
        if self.mode != SafetyMode.NORMAL and dd > -0.01 and locked_count == 0:
            self.set_mode(SafetyMode.NORMAL, "conditions_improved")

        return self.mode


# ============================================================================
# Portfolio Admission
# ============================================================================

class PortfolioAdmission:
    """
    Gates new entries into the portfolio.
    Rules:
        - Max 6 active positions
        - Max 2 new entries per cycle
        - Max 4 new entries per day
        - Category caps (e.g., max 2 meme, max 3 tech)
        - Replacement margin: new candidate must score 1.15x vs worst held
    """

    def __init__(self, policy: ExecutionPolicy):
        self.policy = policy
        self.entries_today: int = 0
        self.entries_this_cycle: int = 0
        self.last_day: Optional[object] = None

    def reset_cycle(self):
        self.entries_this_cycle = 0

    def reset_day(self, today):
        if self.last_day != today:
            self.last_day = today
            self.entries_today = 0

    def can_admit(self, candidate_signal: TradeSignal, candidate_score: float,
                  campaigns: Dict[str, SymbolCampaign],
                  snapshot: BrokerSnapshot) -> Tuple[bool, str]:
        """Check if a new candidate can be admitted to the portfolio."""
        # Cycle limit
        if self.entries_this_cycle >= self.policy.max_new_per_cycle:
            return False, f"max_new_per_cycle({self.entries_this_cycle})"

        # Day limit
        if self.entries_today >= self.policy.max_new_per_day:
            return False, f"max_new_per_day({self.entries_today})"

        # Active positions count
        active_count = snapshot.n_active_positions
        if active_count >= self.policy.max_concurrent_positions:
            # Check if replacement is possible
            return self._check_replacement(candidate_score, campaigns)

        # Category cap
        category = candidate_signal.category
        cap = self.policy.category_caps.get(category, self.policy.max_concurrent_positions)
        cat_count = sum(1 for c in campaigns.values()
                        if c.state in ("BUILDING", "ACTIVE", "PROTECT_PROFIT")
                        and c.category == category)
        if cat_count >= cap:
            return False, f"category_cap({category}={cat_count}/{cap})"

        return True, "admitted"

    def _check_replacement(self, candidate_score: float,
                           campaigns: Dict[str, SymbolCampaign]) -> Tuple[bool, str]:
        """Check if candidate can replace worst held position."""
        active_campaigns = [c for c in campaigns.values()
                           if c.state in ("ACTIVE", "REVALIDATE_LOSER")]
        if not active_campaigns:
            return False, "max_positions_no_replaceable"

        worst = min(active_campaigns, key=lambda c: c.hold_score)
        threshold = worst.hold_score * self.policy.replacement_margin

        if candidate_score > threshold:
            return True, f"replaces({worst.symbol},score={worst.hold_score:.2f})"
        return False, f"replacement_margin_not_met({candidate_score:.2f}<{threshold:.2f})"

    def record_admission(self):
        self.entries_this_cycle += 1
        self.entries_today += 1


# ============================================================================
# Contradiction Detector
# ============================================================================

def resolve_contradictions(symbol: str, campaign: SymbolCampaign,
                           broker_state: BrokerSymbolState) -> Optional[str]:
    """
    Detects impossible states and returns action needed.
    Returns None if no contradiction.
    """
    # Buy orders open during exit (check FIRST - more specific than generic buys-with-target-zero)
    if campaign.state == "EXITING" and broker_state.open_buy_order_ids:
        return "lock_buy_during_exit"

    # Buys open but target is zero
    if broker_state.open_buy_order_ids and campaign.target_qty == 0:
        return "cancel_buys_target_zero"

    # Simultaneous buys and sells
    if broker_state.open_buy_order_ids and broker_state.open_sell_order_ids:
        return "lock_conflicting_orders"

    # Position exists but campaign is LOCKED_ERROR with target=0
    if campaign.state == "LOCKED_ERROR" and broker_state.position_qty > 0:
        # Exit already in flight — not a contradiction
        if broker_state.open_sell_order_ids and campaign.target_qty == 0:
            return None
        return "locked_with_position"

    return None


# ============================================================================
# Execution Reconciler
# ============================================================================

class ExecutionReconciler:
    """
    Reconciles broker truth to campaign targets.
    Does NOT own signal logic -- only manages orders.
    """

    def __init__(self, broker, policy: ExecutionPolicy, eastern_tz):
        self.broker = broker
        self.policy = policy
        self.eastern = eastern_tz
        self.audit_log: List[dict] = []
        self._force_exit_logged: Set[str] = set()

    def _log(self, symbol: str, action: str, details: str):
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "action": action,
            "details": details,
        }
        self.audit_log.append(entry)
        print(f"  [EXEC] {symbol}: {action} -- {details}")


    def cancel_stale_orders(self, snapshot: BrokerSnapshot, now_utc: datetime):
        """Cancel any bot-owned orders that exceed their TTL."""
        for order in snapshot.all_open_orders:
            client_id = order.get("client_order_id", "")
            if not is_bot_order(client_id):
                continue

            submitted_at = order.get("submitted_at")
            if submitted_at is None:
                continue

            age_sec = (now_utc - submitted_at).total_seconds()

            # Extract bucket from client_order_id
            parts = client_id.split("|")
            bucket = parts[3] if len(parts) > 3 else "default"
            ttl = self.policy.order_ttl_seconds_by_bucket.get(
                bucket, self.policy.default_order_ttl_seconds)

            if age_sec > ttl:
                order_id = order.get("id", "")
                symbol = order.get("symbol", "?")
                self.broker.cancel_order(order_id)
                self._log(symbol, "CANCEL_STALE",
                          f"order={order_id} age={age_sec:.0f}s > ttl={ttl}s")

    def resolve_all_contradictions(self, campaigns: Dict[str, SymbolCampaign],
                                   snapshot: BrokerSnapshot) -> int:
        """Detect and resolve impossible states. Returns count resolved."""
        resolved = 0
        for symbol, campaign in campaigns.items():
            broker_state = snapshot.get_symbol_state(symbol)
            contradiction = resolve_contradictions(symbol, campaign, broker_state)

            if contradiction is None:
                continue

            resolved += 1
            if contradiction == "cancel_buys_target_zero":
                for oid in broker_state.open_buy_order_ids:
                    self.broker.cancel_order(oid)
                campaign.state = "EXITING"
                self._log(symbol, "RESOLVE", "cancelled buys (target=0)")

            elif contradiction == "lock_conflicting_orders":
                self.broker.cancel_all_orders_for_symbol(symbol)
                campaign.state = "LOCKED_ERROR"
                campaign.locked_reason = "simultaneous_buy_sell"
                self._log(symbol, "LOCK", "conflicting buy+sell orders")

            elif contradiction == "lock_buy_during_exit":
                self.broker.cancel_all_orders_for_symbol(symbol)
                campaign.state = "LOCKED_ERROR"
                campaign.locked_reason = "buy_during_exit"
                self._log(symbol, "LOCK", "buy orders open during EXITING")

            elif contradiction == "locked_with_position":
                # Locked blocks new entries but must still flatten the position
                for oid in broker_state.open_buy_order_ids:
                    self.broker.cancel_order(oid)
                action = self.reconcile_symbol(symbol, campaign, broker_state)
                if symbol not in self._force_exit_logged:
                    self._force_exit_logged.add(symbol)
                    self._log(symbol, "FORCE_EXIT",
                              f"locked session, flattening position ({action})")
                if action == "reducing_exposure":
                    resolved -= 1  # sell submitted; not blocking

        return resolved


    def reconcile_symbol(self, symbol: str, campaign: SymbolCampaign,
                         broker_state: BrokerSymbolState) -> str:
        """
        Core reconciliation: move effective_exposure toward target_qty.
        Returns action taken.
        """
        desired = campaign.target_qty
        effective = broker_state.effective_exposure

        # Already aligned
        if desired == effective:
            if campaign.state == "EXITING" and broker_state.position_qty == 0:
                campaign.state = "FLAT"
                campaign.cooldown_until = datetime.utcnow() + timedelta(
                    seconds=self.policy.cooldown_seconds)
                campaign.transition_to("COOLDOWN_BLOCKED")
            elif campaign.state == "BUILDING" and broker_state.position_qty >= desired:
                campaign.transition_to("ACTIVE")
                campaign.entry_price = broker_state.avg_entry_price
            return "in_sync"

        # Need to reduce exposure
        if desired < effective:
            return self._reduce_exposure(symbol, campaign, broker_state, desired)

        # Need to increase exposure
        if desired > effective:
            return self._increase_exposure(symbol, campaign, broker_state, desired)

        return "no_action"

    def _reduce_exposure(self, symbol: str, campaign: SymbolCampaign,
                         broker_state: BrokerSymbolState, desired_qty: float) -> str:
        """Cancel pending buys BEFORE selling filled shares (prevents AMD bug)."""
        # 1. Cancel all open buys first
        if broker_state.open_buy_order_ids:
            for order_id in broker_state.open_buy_order_ids:
                self.broker.cancel_order(order_id)
            self._log(symbol, "CANCEL_BUYS",
                      f"cancelled {len(broker_state.open_buy_order_ids)} buys before reducing")
            adjusted_position = broker_state.position_qty
        else:
            adjusted_position = broker_state.position_qty

        # 2. Sell only if filled position exceeds desired target
        if adjusted_position > desired_qty:
            sell_qty = adjusted_position - desired_qty
            client_id = generate_client_order_id(
                symbol, campaign.campaign_id, "SELL", campaign.setup_fingerprint or "flat")
            now_et = datetime.now(self.eastern)
            self.broker.submit_exit_order(
                symbol=symbol, qty=sell_qty, client_order_id=client_id,
                extended_hours=self.policy.allow_extended_hours, now_et=now_et)
            self._log(symbol, "SELL",
                      f"qty={sell_qty} (pos={adjusted_position} -> target={desired_qty})")

        if desired_qty == 0:
            if campaign.state != "LOCKED_ERROR":
                campaign.transition_to("EXITING")
        else:
            campaign.transition_to("REDUCING")

        return "reducing_exposure"


    def _increase_exposure(self, symbol: str, campaign: SymbolCampaign,
                           broker_state: BrokerSymbolState, desired_qty: float) -> str:
        """Submit buy to move toward target. Respects state guards."""
        if campaign.state in ("EXITING", "LOCKED_ERROR", "REDUCING", "COOLDOWN_BLOCKED",
                              "REVALIDATE_LOSER"):
            return f"entry_blocked({campaign.state})"

        if campaign.state == "PROTECT_PROFIT":
            return "entry_blocked(PROTECT_PROFIT_no_adding)"

        if broker_state.open_sell_order_ids:
            return "entry_blocked_exit_open"

        needed_qty = desired_qty - broker_state.effective_exposure
        if needed_qty <= 0:
            return "no_buy_needed"

        # Don't stack buys
        if broker_state.open_buy_order_ids:
            return "buy_already_pending"

        # Session window check
        now_et = datetime.now(self.eastern)
        in_window, reason = is_in_trading_window(now_et, self.policy)
        if not in_window:
            return f"outside_window({reason})"

        client_id = generate_client_order_id(
            symbol, campaign.campaign_id, "BUY", campaign.setup_fingerprint or "new")

        success = self.broker.submit_entry_order(
            symbol=symbol, qty=needed_qty, client_order_id=client_id,
            invalidation_price=campaign.invalidation_price,
            extended_hours=self.policy.allow_extended_hours, now_et=now_et)

        if not success:
            campaign.target_qty = 0
            campaign.state = "FLAT"
            self._log(symbol, "ORDER_REJECTED",
                      f"qty={needed_qty} failed, resetting to FLAT")
            return "order_rejected"

        campaign.transition_to("BUILDING")
        self._log(symbol, "BUY",
                  f"qty={needed_qty} (effective={broker_state.effective_exposure} -> target={desired_qty})")
        return "increasing_exposure"

    def reconcile_all(self, campaigns: Dict[str, SymbolCampaign],
                      snapshot: BrokerSnapshot) -> Dict[str, str]:
        """Reconcile all campaigns. Returns {symbol: action_taken}."""
        results = {}
        for symbol, campaign in campaigns.items():
            broker_state = snapshot.get_symbol_state(symbol)
            action = self.reconcile_symbol(symbol, campaign, broker_state)
            results[symbol] = action
        return results


# ============================================================================
# Alpaca Broker Adapter
# ============================================================================

class AlpacaBrokerAdapter:
    """
    Thin wrapper around Alpaca trading client.
    All order operations go through here. Single point of broker interaction.
    """

    def __init__(self, trading_client, eastern_tz):
        self.client = trading_client
        self.eastern = eastern_tz

    def fetch_snapshot(self) -> BrokerSnapshot:
        """Build complete broker truth from Alpaca in 2-3 API calls."""
        from alpaca.trading.enums import OrderSide

        # Account
        account = self.client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)

        # Positions
        raw_positions = self.client.get_all_positions()
        positions: Dict[str, BrokerSymbolState] = {}
        for p in raw_positions:
            positions[p.symbol] = BrokerSymbolState(
                symbol=p.symbol,
                position_qty=int(p.qty),
                avg_entry_price=float(p.avg_entry_price) if p.avg_entry_price else None,
                pending_buy_qty=0,
                pending_sell_qty=0,
            )

        # Open orders
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        raw_orders = self.client.get_orders(req)

        all_open_orders = []
        for o in raw_orders:
            order_dict = {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": str(o.side),
                "qty": int(o.qty) if o.qty else 0,
                "filled_qty": int(o.filled_qty) if o.filled_qty else 0,
                "client_order_id": o.client_order_id or "",
                "submitted_at": o.submitted_at,
                "status": str(o.status),
            }
            all_open_orders.append(order_dict)

            remaining_qty = order_dict["qty"] - order_dict["filled_qty"]
            sym = o.symbol

            if sym not in positions:
                positions[sym] = BrokerSymbolState(
                    symbol=sym, position_qty=0, avg_entry_price=None,
                    pending_buy_qty=0, pending_sell_qty=0)

            bs = positions[sym]
            if str(o.side) == "OrderSide.BUY" or "buy" in str(o.side).lower():
                bs.pending_buy_qty += remaining_qty
                bs.open_buy_order_ids.append(str(o.id))
                bs.open_buy_orders.append(order_dict)
            else:
                bs.pending_sell_qty += remaining_qty
                bs.open_sell_order_ids.append(str(o.id))
                bs.open_sell_orders.append(order_dict)

        return BrokerSnapshot(
            account_equity=equity,
            account_buying_power=buying_power,
            positions=positions,
            all_open_orders=all_open_orders,
        )


    def cancel_order(self, order_id: str):
        """Cancel a single order by ID."""
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception as e:
            if "not found" not in str(e).lower() and "not cancelable" not in str(e).lower():
                print(f"    cancel_order failed ({order_id}): {e}")

    def cancel_all_orders_for_symbol(self, symbol: str):
        """Cancel all open orders for a symbol."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = self.client.get_orders(req)
            for o in orders:
                self.cancel_order(str(o.id))
        except Exception as e:
            print(f"    cancel_all failed ({symbol}): {e}")

    def submit_entry_order(self, symbol: str, qty: float, client_order_id: str,
                           invalidation_price: Optional[float] = None,
                           extended_hours: bool = False, now_et: datetime = None) -> bool:
        """Submit entry order with fractional share support. Returns True on success."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        try:
            # Use fractional qty directly (Alpaca supports decimal qty for market orders)
            oreq = MarketOrderRequest(
                symbol=symbol, qty=round(qty, 4), side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id)
            self.client.submit_order(oreq)
            return True
        except Exception as e:
            print(f"    submit_entry failed ({symbol} qty={qty}): {e}")
            return False

    def submit_exit_order(self, symbol: str, qty: float, client_order_id: str,
                          extended_hours: bool = False, now_et: datetime = None) -> bool:
        """Submit exit order with fractional share support. Returns True on success."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        try:
            oreq = MarketOrderRequest(
                symbol=symbol, qty=round(qty, 4), side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id)
            self.client.submit_order(oreq)
            return True
        except Exception as e:
            print(f"    submit_exit failed ({symbol} qty={qty}): {e}")
            return False


# ============================================================================
# Campaign Book
# ============================================================================

class CampaignBook:
    """Manages all SymbolCampaigns. Single source of target state."""

    def __init__(self, symbols: List[str]):
        self.campaigns: Dict[str, SymbolCampaign] = {
            sym: SymbolCampaign(symbol=sym) for sym in symbols
        }

    def get(self, symbol: str) -> SymbolCampaign:
        if symbol not in self.campaigns:
            self.campaigns[symbol] = SymbolCampaign(symbol=symbol)
        return self.campaigns[symbol]

    def items(self):
        return self.campaigns.items()

    def active_campaigns(self) -> List[SymbolCampaign]:
        return [c for c in self.campaigns.values()
                if c.state in ("BUILDING", "ACTIVE", "PROTECT_PROFIT", "REVALIDATE_LOSER")]

    def sync_from_broker(self, snapshot: BrokerSnapshot):
        """Update campaign states based on broker truth."""
        for symbol, campaign in self.campaigns.items():
            bs = snapshot.get_symbol_state(symbol)

            # Position filled and target met
            if campaign.state == "BUILDING" and bs.position_qty >= campaign.target_qty > 0:
                campaign.transition_to("ACTIVE")
                campaign.entry_price = bs.avg_entry_price

            # Position fully exited
            if campaign.state == "EXITING" and bs.position_qty == 0 and bs.pending_sell_qty == 0:
                campaign.cooldown_until = datetime.utcnow() + timedelta(
                    seconds=self.policy_ref.cooldown_seconds if hasattr(self, 'policy_ref') else 300)
                campaign.transition_to("COOLDOWN_BLOCKED")
                campaign.target_qty = 0

            # Cooldown expired
            if campaign.state == "COOLDOWN_BLOCKED":
                if campaign.cooldown_until and datetime.utcnow() >= campaign.cooldown_until:
                    campaign.reset_flat()

            # Unexpected: position exists but campaign thinks FLAT
            if campaign.state == "FLAT" and bs.position_qty > 0:
                campaign.state = "ACTIVE"
                campaign.target_qty = bs.position_qty
                campaign.entry_price = bs.avg_entry_price

        # Also add any broker positions not in campaign book
        for sym, bs in snapshot.positions.items():
            if bs.position_qty > 0 and sym not in self.campaigns:
                c = SymbolCampaign(symbol=sym, state="ACTIVE",
                                   target_qty=bs.position_qty,
                                   entry_price=bs.avg_entry_price)
                self.campaigns[sym] = c


    def set_target(self, symbol: str, signal: TradeSignal, target_qty: int,
                   invalidation_price: Optional[float] = None):
        """Apply a SET_TARGET decision."""
        c = self.get(symbol)
        c.target_qty = target_qty
        c.max_qty = max(c.max_qty, target_qty)
        c.bucket_id = signal.bucket_id
        c.setup_fingerprint = signal.setup_fingerprint
        c.category = signal.category
        c.invalidation_price = invalidation_price or signal.invalidation_price
        c.last_signal_time = datetime.utcnow()
        c.confirmations += 1
        if c.state == "FLAT":
            c.transition_to("WATCH")
        if c.state == "WATCH":
            c.transition_to("BUILDING")

    def set_target_zero(self, symbol: str, reason: str):
        """Apply a TARGET_ZERO decision."""
        c = self.get(symbol)
        c.target_qty = 0
        c.blocked_setup_fingerprint = c.setup_fingerprint
        c.last_exit_time = datetime.utcnow()
        c.last_exit_reason = reason
        if c.state not in ("EXITING", "LOCKED_ERROR"):
            c.transition_to("EXITING")

    def lock(self, symbol: str, reason: str):
        """Lock a symbol from trading."""
        c = self.get(symbol)
        c.state = "LOCKED_ERROR"
        c.locked_reason = reason
        c.target_qty = 0


# ============================================================================
# Session Policy Helpers
# ============================================================================

def is_in_trading_window(now_et: datetime, policy: ExecutionPolicy) -> Tuple[bool, str]:
    """Check if we're within the allowed trading window."""
    time_str = now_et.strftime("%H:%M")

    if time_str < policy.no_new_entries_before:
        return False, f"before_entry_window({time_str}<{policy.no_new_entries_before})"

    if time_str > policy.no_new_entries_after:
        return False, f"after_entry_window({time_str}>{policy.no_new_entries_after})"

    return True, "in_window"


def should_cancel_premarket_orders(now_et: datetime, policy: ExecutionPolicy) -> bool:
    """Returns True if we should cancel all premarket orders."""
    time_str = now_et.strftime("%H:%M")
    return time_str >= policy.cancel_premarket_orders_at and time_str < "09:30"


# ============================================================================
# Decision Engine (enhanced with loss governance + profit protection)
# ============================================================================

def evaluate_signal(signal: TradeSignal, campaign: SymbolCampaign,
                    broker_state: BrokerSymbolState, market_state: MarketState,
                    costs: CostEstimate, account_equity: float, price: float,
                    policy: ExecutionPolicy, buying_power: float = None,
                    loss_governor: LossGovernor = None) -> Decision:
    """
    Central decision: should this signal change the campaign target?
    Returns a Decision (SET_TARGET, TARGET_ZERO, LOCK, NO_ACTION).
    """
    # Expired signal
    if signal.is_expired:
        return Decision(action="NO_ACTION", reason="signal_expired")

    # Campaign locked
    if campaign.state == "LOCKED_ERROR":
        return Decision(action="NO_ACTION", reason=f"locked({campaign.locked_reason})")

    # Cooldown blocked
    if campaign.state == "COOLDOWN_BLOCKED":
        return Decision(action="NO_ACTION", reason="cooldown_blocked")

    # Direction: flat = exit
    if signal.direction == "flat":
        if campaign.target_qty > 0 or broker_state.effective_exposure > 0:
            return Decision(action="TARGET_ZERO", reason=signal.reason or "signal_flat")
        return Decision(action="NO_ACTION", reason="already_flat")

    # Direction: long
    # Check profit protection (no adding above +0.75%)
    if loss_governor and campaign.entry_price and price > 0:
        profit_action = loss_governor.evaluate_profit_protection(campaign, price)
        if profit_action == "no_adding" and campaign.state in ("ACTIVE", "PROTECT_PROFIT"):
            return Decision(action="NO_ACTION", reason="profit_no_adding(+0.75%)")

    # 1. No-trade-zone gate
    ntz_ok, ntz_reason = passes_no_trade_zone(signal, costs, policy)
    if not ntz_ok:
        return Decision(action="NO_ACTION", reason=ntz_reason)

    # 2. Re-entry qualification
    re_ok, re_reason = passes_reentry_gate(signal, campaign, market_state, policy)
    if not re_ok:
        return Decision(action="NO_ACTION", reason=re_reason)

    # 3. Compute target (with buying power cap)
    target = compute_target_qty(signal, account_equity, price, market_state.atr, policy,
                                buying_power=buying_power)
    if target <= 0:
        return Decision(action="NO_ACTION", reason="target_qty_zero")

    return Decision(action="SET_TARGET", target_qty=target, reason=f"entry({signal.bucket_id})")


# ============================================================================
# Main Controller Loop (V14_2_1 architecture)
# ============================================================================

def run_controller_loop(signal_engine, broker_adapter: AlpacaBrokerAdapter,
                        campaign_book: CampaignBook, policy: ExecutionPolicy,
                        eastern_tz, check_interval: int = 45):
    """
    V14_2_1 Loss-Governed Active Portfolio Controller main loop.

    Loop order (spec-mandated):
        1. broker truth
        2. rebuild ledgers
        3. detect contradictions
        4. cancel stale
        5. refresh
        6. rescore positions
        7. recompute targets (loss governance + profit protection)
        8. reduce/exit
        9. refresh
        10. block if unresolved
        11. generate candidates
        12. admit
        13. assign targets
        14. submit buys
        15. audit
    """
    import pytz

    reconciler = ExecutionReconciler(broker_adapter, policy, eastern_tz)
    loss_governor = LossGovernor(policy)
    safety_manager = SafetyModeManager()
    hold_scorer = HoldScorer()
    admission = PortfolioAdmission(policy)
    campaign_book.policy_ref = policy

    trades_today = 0
    day_start_equity = None
    current_day = None

    print(f"\n{'=' * 70}")
    print(f"  Execution Controller {VERSION} - Loss-Governed Active Portfolio")
    print(f"  Symbols: {len(campaign_book.campaigns)}")
    print(f"  Policy: entries {policy.no_new_entries_before}-{policy.no_new_entries_after} ET")
    print(f"  Max positions: {policy.max_concurrent_positions}")
    print(f"  Loss thresholds: revalidate={LOSS_REVALIDATE_PCT*100:.2f}% "
          f"zero={LOSS_TARGET_ZERO_PCT*100:.2f}% lock={LOSS_LOCK_SESSION_PCT*100:.2f}%")
    print(f"  Safety mode: {safety_manager.mode.value}")
    print(f"{'=' * 70}\n")

    while True:
        try:
            now_et = datetime.now(eastern_tz)
            now_utc = datetime.utcnow()

            # Daily reset
            today = now_et.date()
            if current_day != today:
                current_day = today
                trades_today = 0
                admission.reset_day(today)
                loss_governor.session_locked_symbols.clear()
                snapshot = broker_adapter.fetch_snapshot()
                day_start_equity = snapshot.account_equity
                print(f"\n  -- New day: {today} | equity=${day_start_equity:,.2f}")

            # Weekend check
            if now_et.weekday() >= 5:
                time.sleep(300)
                continue

            # ─── 1. BROKER TRUTH ─────────────────────────────────────────
            snapshot = broker_adapter.fetch_snapshot()

            # ─── 2. REBUILD LEDGERS ──────────────────────────────────────
            campaign_book.sync_from_broker(snapshot)

            # Safety mode evaluation
            safety_manager.evaluate_portfolio_health(
                snapshot, campaign_book.campaigns, day_start_equity or snapshot.account_equity)

            if safety_manager.mode == SafetyMode.MANUAL_REVIEW_REQUIRED:
                print(f"  [{now_et:%H:%M:%S}] MANUAL REVIEW REQUIRED - halted")
                time.sleep(300)
                continue


            # ─── 3. DETECT CONTRADICTIONS ────────────────────────────────
            contradictions = reconciler.resolve_all_contradictions(
                campaign_book.campaigns, snapshot)

            # ─── 4. CANCEL STALE ─────────────────────────────────────────
            reconciler.cancel_stale_orders(snapshot, now_utc)

            # Pre-market order sweep
            if should_cancel_premarket_orders(now_et, policy):
                for order in snapshot.all_open_orders:
                    if is_bot_order(order.get("client_order_id", "")):
                        broker_adapter.cancel_order(order["id"])
                time.sleep(check_interval)
                continue

            # ─── 5. REFRESH ──────────────────────────────────────────────
            snapshot = broker_adapter.fetch_snapshot()

            # ─── 6. RESCORE POSITIONS ────────────────────────────────────
            for symbol, campaign in campaign_book.items():
                if campaign.state in ("ACTIVE", "BUILDING", "PROTECT_PROFIT", "REVALIDATE_LOSER"):
                    bs = snapshot.get_symbol_state(symbol)
                    price = bs.avg_entry_price or 0.0
                    costs = signal_engine.estimate_costs(symbol)
                    hold_scorer.score(campaign, current_price=price,
                                     spread_bps=costs.spread_bps)
                    campaign.bars_held += 1

            # ─── 7. RECOMPUTE TARGETS (loss governance + profit protection) ──
            for symbol, campaign in campaign_book.items():
                if campaign.state in ("ACTIVE", "BUILDING", "PROTECT_PROFIT", "REVALIDATE_LOSER"):
                    bs = snapshot.get_symbol_state(symbol)
                    ms = signal_engine.get_market_state(symbol)
                    current_price = ms.last_price if ms.last_price > 0 else (bs.avg_entry_price or 0.0)

                    # Loss governance
                    loss_action = loss_governor.evaluate(campaign, current_price)
                    if loss_action:
                        has_fresh = (campaign.last_signal_time and
                                     (now_utc - campaign.last_signal_time).total_seconds() < 120)
                        new_state = loss_governor.apply_loss_action(
                            campaign, loss_action, has_fresh_confirm=has_fresh)
                        if new_state in ("EXITING", "LOCKED_ERROR"):
                            continue

                    # Profit protection
                    profit_action = loss_governor.evaluate_profit_protection(campaign, current_price)
                    if profit_action:
                        loss_governor.apply_profit_action(campaign, profit_action)

            # ─── 8. REDUCE/EXIT ──────────────────────────────────────────
            # Handle campaigns that need reducing/exiting
            for symbol, campaign in campaign_book.items():
                if campaign.state in ("EXITING", "REDUCING"):
                    bs = snapshot.get_symbol_state(symbol)
                    reconciler.reconcile_symbol(symbol, campaign, bs)
                elif (campaign.state == "LOCKED_ERROR" and campaign.target_qty == 0):
                    bs = snapshot.get_symbol_state(symbol)
                    if bs.position_qty > 0 or bs.open_buy_order_ids:
                        reconciler.reconcile_symbol(symbol, campaign, bs)

            # Liquidate-only mode: exit everything
            if safety_manager.is_liquidate_only():
                for symbol, campaign in campaign_book.items():
                    if campaign.state in ("ACTIVE", "BUILDING", "PROTECT_PROFIT", "REVALIDATE_LOSER"):
                        campaign_book.set_target_zero(symbol, "liquidate_mode")
                time.sleep(check_interval)
                continue


            # ─── 9. REFRESH ──────────────────────────────────────────────
            snapshot = broker_adapter.fetch_snapshot()

            # ─── 10. BLOCK IF UNRESOLVED ─────────────────────────────────
            unresolved = reconciler.resolve_all_contradictions(
                campaign_book.campaigns, snapshot)
            if unresolved > 0:
                safety_manager.set_mode(SafetyMode.ORDER_RECONCILIATION_ONLY,
                                        f"unresolved_contradictions={unresolved}")

            # ─── 11. GENERATE CANDIDATES ─────────────────────────────────
            if not safety_manager.allows_new_entries():
                time.sleep(check_interval)
                continue

            signals: List[TradeSignal] = signal_engine.generate_signals()
            in_window, window_reason = is_in_trading_window(now_et, policy)

            # ─── 12. ADMIT ───────────────────────────────────────────────
            admission.reset_cycle()
            admitted_signals: List[Tuple[TradeSignal, Decision]] = []

            for signal in signals:
                campaign = campaign_book.get(signal.symbol)
                broker_state = snapshot.get_symbol_state(signal.symbol)
                market_state = signal_engine.get_market_state(signal.symbol)
                costs = signal_engine.estimate_costs(signal.symbol)

                decision = evaluate_signal(
                    signal=signal, campaign=campaign, broker_state=broker_state,
                    market_state=market_state, costs=costs,
                    account_equity=snapshot.account_equity,
                    price=market_state.last_price, policy=policy,
                    buying_power=snapshot.account_buying_power,
                    loss_governor=loss_governor)

                # Session window blocks new entries (but allows exits)
                if decision.action == "SET_TARGET" and not in_window:
                    decision = Decision(action="NO_ACTION", reason=window_reason)

                # Exit decisions always go through
                if decision.action == "TARGET_ZERO":
                    campaign_book.set_target_zero(signal.symbol, decision.reason)
                    continue

                if decision.action != "SET_TARGET":
                    continue

                # Admission gate
                can_admit, admit_reason = admission.can_admit(
                    signal, decision.target_qty, campaign_book.campaigns, snapshot)
                if not can_admit:
                    continue

                admitted_signals.append((signal, decision))


            # ─── 13. ASSIGN TARGETS ──────────────────────────────────────
            for signal, decision in admitted_signals:
                campaign_book.set_target(signal.symbol, signal, decision.target_qty,
                                        signal.invalidation_price)
                admission.record_admission()
                trades_today += 1

            # ─── 14. SUBMIT BUYS ─────────────────────────────────────────
            fresh_snapshot = broker_adapter.fetch_snapshot()
            results = reconciler.reconcile_all(campaign_book.campaigns, fresh_snapshot)

            # ─── 15. AUDIT ───────────────────────────────────────────────
            active = sum(1 for r in results.values() if r not in ("in_sync", "no_action"))
            n_pos = fresh_snapshot.n_active_positions
            if active > 0 or int(time.time()) % 120 < check_interval:
                exp_pct = fresh_snapshot.total_exposure_notional / max(fresh_snapshot.account_equity, 1.0)
                print(f"  [{now_et:%H:%M:%S}] pos={n_pos} exposure={exp_pct:.1%} "
                      f"trades={trades_today} safety={safety_manager.mode.value} "
                      f"reconciled={active}")

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print(f"\n  Shutting down {VERSION} execution controller...")
            break
        except Exception as e:
            print(f"  Loop error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(check_interval)


# Backward compatibility alias
run_execution_loop = run_controller_loop


# ============================================================================
# REGRESSION TESTS
# ============================================================================

def _run_regression_tests():
    """
    Full regression test suite for V14_2_1 Loss-Governed Active Portfolio Controller.
    Run with: python execution_controller.py
    """
    import sys
    passed = 0
    failed = 0
    errors = []

    def assert_eq(actual, expected, test_name):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
        else:
            failed += 1
            errors.append(f"  FAIL: {test_name}: got {actual!r}, expected {expected!r}")

    def assert_true(condition, test_name):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            errors.append(f"  FAIL: {test_name}")

    def assert_false(condition, test_name):
        assert_true(not condition, test_name)

    print(f"\n{'=' * 70}")
    print(f"  V14_2_1 Regression Tests")
    print(f"{'=' * 70}\n")

    # ─── Test 1: Symbol States ────────────────────────────────────────
    print("  [1] Symbol States...")
    c = SymbolCampaign(symbol="TEST")
    assert_eq(c.state, "FLAT", "initial state is FLAT")
    assert_true(c.transition_to("WATCH"), "FLAT->WATCH allowed")
    assert_eq(c.state, "WATCH", "state is WATCH")
    assert_true(c.transition_to("BUILDING"), "WATCH->BUILDING allowed")
    assert_true(c.transition_to("ACTIVE"), "BUILDING->ACTIVE allowed")
    assert_true(c.transition_to("PROTECT_PROFIT"), "ACTIVE->PROTECT_PROFIT allowed")
    assert_true(c.transition_to("REDUCING"), "PROTECT_PROFIT->REDUCING allowed")
    assert_true(c.transition_to("EXITING"), "REDUCING->EXITING allowed")


    # ─── Test 2: Forbidden Transitions ────────────────────────────────
    print("  [2] Forbidden Transitions...")
    c2 = SymbolCampaign(symbol="TEST2", state="EXITING")
    assert_false(c2.transition_to("BUILDING"), "EXITING->BUILDING forbidden")
    assert_eq(c2.state, "EXITING", "state unchanged after forbidden")

    c3 = SymbolCampaign(symbol="TEST3", state="REDUCING")
    assert_false(c3.transition_to("BUILDING"), "REDUCING->BUILDING forbidden")

    c4 = SymbolCampaign(symbol="TEST4", state="LOCKED_ERROR")
    assert_false(c4.transition_to("BUILDING"), "LOCKED_ERROR->BUILDING forbidden")
    assert_false(c4.transition_to("ACTIVE"), "LOCKED_ERROR->ACTIVE forbidden")
    assert_false(c4.transition_to("WATCH"), "LOCKED_ERROR->WATCH forbidden")

    c5 = SymbolCampaign(symbol="TEST5", state="COOLDOWN_BLOCKED")
    assert_false(c5.transition_to("BUILDING"), "COOLDOWN_BLOCKED->BUILDING forbidden")
    assert_false(c5.transition_to("ACTIVE"), "COOLDOWN_BLOCKED->ACTIVE forbidden")

    c6 = SymbolCampaign(symbol="TEST6", state="EXITING")
    assert_false(c6.transition_to("ACTIVE"), "EXITING->ACTIVE forbidden")

    # ─── Test 3: Client Order ID Format ──────────────────────────────
    print("  [3] Client Order ID Format...")
    cid = generate_client_order_id("AAPL", "abc12345", "BUY", "fp123")
    assert_true(cid.startswith("BOT|V14_2_1|AAPL|abc12345|BUY|"), "order ID format correct")
    parts = cid.split("|")
    assert_eq(len(parts), 7, "order ID has 7 parts")
    assert_eq(parts[0], "BOT", "part[0]=BOT")
    assert_eq(parts[1], "V14_2_1", "part[1]=V14_2_1")
    assert_eq(parts[2], "AAPL", "part[2]=AAPL")
    assert_eq(parts[3], "abc12345", "part[3]=campaign_id")
    assert_eq(parts[4], "BUY", "part[4]=SIDE")

    # ─── Test 4: is_bot_order (both prefixes) ────────────────────────
    print("  [4] is_bot_order (dual prefix)...")
    assert_true(is_bot_order("BOT|V14_2_1|AAPL|abc|BUY|20240101T120000|deadbeef"),
                "V14_2_1 prefix recognized")
    assert_true(is_bot_order("BOT|V15|AAPL|B4|20240101T120000|BUY|abcdef"),
                "V15 prefix recognized")
    assert_false(is_bot_order("MANUAL|AAPL|BUY"), "non-bot order rejected")
    assert_false(is_bot_order(""), "empty string rejected")
    assert_false(is_bot_order(None), "None rejected")


    # ─── Test 5: Loss Governor Thresholds ─────────────────────────────
    print("  [5] Loss Governor Thresholds...")
    lg = LossGovernor(ExecutionPolicy())

    # -1.50% -> revalidate
    c_loss = SymbolCampaign(symbol="LOSS", state="ACTIVE", entry_price=100.0)
    result = lg.evaluate(c_loss, 98.50)  # exactly -1.50%
    assert_eq(result, "revalidate", "-1.50% triggers revalidate")

    # -2.50% -> fresh_confirm
    result = lg.evaluate(c_loss, 97.50)
    assert_eq(result, "fresh_confirm", "-2.50% triggers fresh_confirm")

    # -4.00% -> target_zero
    result = lg.evaluate(c_loss, 96.00)
    assert_eq(result, "target_zero", "-4.00% triggers target_zero")

    # -6.00% -> lock_session
    result = lg.evaluate(c_loss, 94.00)
    assert_eq(result, "lock_session", "-6.00% triggers lock_session")
    assert_true("LOSS" in lg.session_locked_symbols, "symbol added to session locked")

    # No loss -> None
    c_ok = SymbolCampaign(symbol="OK", state="ACTIVE", entry_price=100.0)
    result = lg.evaluate(c_ok, 100.50)
    assert_eq(result, None, "no loss -> None")

    # FLAT state -> None (skipped)
    c_flat = SymbolCampaign(symbol="FLAT_TEST", state="FLAT", entry_price=100.0)
    result = lg.evaluate(c_flat, 90.0)
    assert_eq(result, None, "FLAT state skipped by loss governor")

    # ─── Test 6: Profit Protection Thresholds ────────────────────────
    print("  [6] Profit Protection Thresholds...")
    lg2 = LossGovernor(ExecutionPolicy())

    c_profit = SymbolCampaign(symbol="WIN", state="ACTIVE", entry_price=100.0)

    # +0.75% -> no_adding
    result = lg2.evaluate_profit_protection(c_profit, 100.75)
    assert_eq(result, "no_adding", "+0.75% triggers no_adding")

    # +1.25% -> protect_profit
    result = lg2.evaluate_profit_protection(c_profit, 101.25)
    assert_eq(result, "protect_profit", "+1.25% triggers protect_profit")

    # +2.00% -> trim_trail
    result = lg2.evaluate_profit_protection(c_profit, 102.00)
    assert_eq(result, "trim_trail", "+2.00% triggers trim_trail")

    # Below threshold -> None
    result = lg2.evaluate_profit_protection(c_profit, 100.50)
    assert_eq(result, None, "below +0.75% -> None")


    # ─── Test 7: Loss Governor Apply Actions ─────────────────────────
    print("  [7] Loss Governor Apply Actions...")
    lg3 = LossGovernor(ExecutionPolicy())

    # lock_session
    c_lock = SymbolCampaign(symbol="LOCK", state="ACTIVE", target_qty=100)
    new_state = lg3.apply_loss_action(c_lock, "lock_session")
    assert_eq(new_state, "LOCKED_ERROR", "lock_session -> LOCKED_ERROR")
    assert_eq(c_lock.target_qty, 0, "locked target=0")

    # target_zero
    c_tz = SymbolCampaign(symbol="TZ", state="ACTIVE", target_qty=100)
    new_state = lg3.apply_loss_action(c_tz, "target_zero")
    assert_eq(new_state, "EXITING", "target_zero -> EXITING")
    assert_eq(c_tz.target_qty, 0, "target_zero target=0")

    # fresh_confirm without confirm
    c_fc = SymbolCampaign(symbol="FC", state="ACTIVE", target_qty=100)
    new_state = lg3.apply_loss_action(c_fc, "fresh_confirm", has_fresh_confirm=False)
    assert_eq(new_state, "EXITING", "fresh_confirm(no confirm) -> EXITING")

    # fresh_confirm with confirm
    c_fc2 = SymbolCampaign(symbol="FC2", state="ACTIVE", target_qty=100)
    new_state = lg3.apply_loss_action(c_fc2, "fresh_confirm", has_fresh_confirm=True)
    assert_eq(new_state, "ACTIVE", "fresh_confirm(with confirm) -> stays ACTIVE")
    assert_eq(c_fc2.target_qty, 100, "fresh_confirm(with confirm) target unchanged")

    # revalidate
    c_rv = SymbolCampaign(symbol="RV", state="ACTIVE", target_qty=100)
    new_state = lg3.apply_loss_action(c_rv, "revalidate")
    assert_eq(c_rv.state, "REVALIDATE_LOSER", "revalidate -> REVALIDATE_LOSER")

    # ─── Test 8: Profit Protection Apply Actions ─────────────────────
    print("  [8] Profit Protection Apply Actions...")
    lg4 = LossGovernor(ExecutionPolicy())

    # trim_trail halves position
    c_trim = SymbolCampaign(symbol="TRIM", state="ACTIVE", target_qty=100)
    lg4.apply_profit_action(c_trim, "trim_trail")
    assert_eq(c_trim.target_qty, 50, "trim_trail halves target")
    assert_eq(c_trim.state, "REDUCING", "trim_trail -> REDUCING")

    # protect_profit
    c_pp = SymbolCampaign(symbol="PP", state="ACTIVE", target_qty=100)
    lg4.apply_profit_action(c_pp, "protect_profit")
    assert_eq(c_pp.state, "PROTECT_PROFIT", "protect_profit -> PROTECT_PROFIT")
    assert_eq(c_pp.target_qty, 100, "protect_profit target unchanged")


    # ─── Test 9: Portfolio Admission ─────────────────────────────────
    print("  [9] Portfolio Admission...")
    policy = ExecutionPolicy()
    adm = PortfolioAdmission(policy)
    adm.reset_day("2024-01-01")

    sig = TradeSignal(symbol="NEW", bucket_id="B4", direction="long", score=0.8,
                      expected_return_bps=10.0, setup_fingerprint="fp1",
                      generated_at=datetime.utcnow(), ttl_seconds=90, category="tech")

    # Empty portfolio -> admit
    campaigns = {}
    snap = BrokerSnapshot(account_equity=100000, account_buying_power=50000)
    can, reason = adm.can_admit(sig, 0.8, campaigns, snap)
    assert_true(can, "empty portfolio admits")

    # Max per cycle
    adm.entries_this_cycle = 3
    can, reason = adm.can_admit(sig, 0.8, campaigns, snap)
    assert_false(can, "max_new_per_cycle blocks")
    assert_true("max_new_per_cycle" in reason, "reason mentions cycle limit")

    # Max per day
    adm.entries_this_cycle = 0
    adm.entries_today = 6
    can, reason = adm.can_admit(sig, 0.8, campaigns, snap)
    assert_false(can, "max_new_per_day blocks")

    # Category cap
    adm.entries_today = 0
    campaigns_cat = {
        f"TECH{i}": SymbolCampaign(symbol=f"TECH{i}", state="ACTIVE", category="tech")
        for i in range(3)
    }
    can, reason = adm.can_admit(sig, 0.8, campaigns_cat, snap)
    assert_false(can, "category cap blocks")
    assert_true("category_cap" in reason, "reason mentions category")

    # ─── Test 10: Replacement Margin ─────────────────────────────────
    print("  [10] Replacement Margin...")
    adm2 = PortfolioAdmission(policy)
    adm2.reset_day("2024-01-01")

    # Full portfolio with weak position
    full_campaigns = {}
    for i in range(8):
        c = SymbolCampaign(symbol=f"SYM{i}", state="ACTIVE", category="general")
        c.hold_score = 0.5 if i > 0 else 0.2  # SYM0 is weak
        full_campaigns[f"SYM{i}"] = c

    full_snap = BrokerSnapshot(account_equity=100000, account_buying_power=50000,
                               positions={f"SYM{i}": BrokerSymbolState(
                                   symbol=f"SYM{i}", position_qty=10, avg_entry_price=50.0,
                                   pending_buy_qty=0, pending_sell_qty=0) for i in range(8)})

    sig_strong = TradeSignal(symbol="STRONG", bucket_id="B5", direction="long", score=0.95,
                             expected_return_bps=15.0, setup_fingerprint="fp2",
                             generated_at=datetime.utcnow(), ttl_seconds=90, category="general")

    # Score of 0.5 > 0.2 * 1.15 = 0.23 -> should replace
    can, reason = adm2.can_admit(sig_strong, 0.5, full_campaigns, full_snap)
    assert_true(can, "strong candidate replaces weak")
    assert_true("replaces" in reason, "reason mentions replacement")

    # Weak candidate can't replace
    can, reason = adm2.can_admit(sig_strong, 0.1, full_campaigns, full_snap)
    assert_false(can, "weak candidate cannot replace")


    # ─── Test 11: Hold Scorer ────────────────────────────────────────
    print("  [11] Hold Scorer...")
    hs = HoldScorer()

    # B5 bucket, all positives
    c_hs = SymbolCampaign(symbol="HS", state="ACTIVE", bucket_id="B5",
                          entry_price=100.0, confirmations=3,
                          trend_aligned=True, volume_confirmed=True, bars_held=5)
    score = hs.score(c_hs, current_price=101.0, spread_bps=2.0)
    assert_true(score > 1.0, f"strong position score > 1.0 (got {score:.2f})")

    # B2 bucket, loser
    c_weak = SymbolCampaign(symbol="WEAK", state="REVALIDATE_LOSER", bucket_id="B2",
                            entry_price=100.0, confirmations=0,
                            trend_aligned=False, volume_confirmed=False, bars_held=50)
    score_weak = hs.score(c_weak, current_price=98.0, spread_bps=10.0)
    assert_true(score_weak < 0, f"weak position score < 0 (got {score_weak:.2f})")
    assert_true(score > score_weak, "strong > weak in scoring")

    # Stale penalty
    c_stale = SymbolCampaign(symbol="STALE", state="ACTIVE", bucket_id="B4",
                             entry_price=100.0, bars_held=30)
    score_stale = hs.score(c_stale, current_price=100.0)
    c_fresh = SymbolCampaign(symbol="FRESH", state="ACTIVE", bucket_id="B4",
                             entry_price=100.0, bars_held=0)
    score_fresh = hs.score(c_fresh, current_price=100.0)
    assert_true(score_fresh > score_stale, "fresh beats stale")

    # ─── Test 12: Safety Mode Manager ────────────────────────────────
    print("  [12] Safety Mode Manager...")
    sm = SafetyModeManager()
    assert_eq(sm.mode, SafetyMode.NORMAL, "initial mode is NORMAL")
    assert_true(sm.allows_new_entries(), "NORMAL allows entries")
    assert_true(sm.allows_exits(), "NORMAL allows exits")

    # Simulate drawdown
    snap_dd = BrokerSnapshot(account_equity=95000, account_buying_power=40000)
    campaigns_sm = {}
    sm.evaluate_portfolio_health(snap_dd, campaigns_sm, 100000.0)
    assert_eq(sm.mode, SafetyMode.LIQUIDATE_ONLY, "-5% -> LIQUIDATE_ONLY")
    assert_false(sm.allows_new_entries(), "LIQUIDATE_ONLY blocks entries")
    assert_true(sm.is_liquidate_only(), "is_liquidate_only correct")

    # 3% drawdown
    sm2 = SafetyModeManager()
    snap_3 = BrokerSnapshot(account_equity=97000, account_buying_power=40000)
    sm2.evaluate_portfolio_health(snap_3, {}, 100000.0)
    assert_eq(sm2.mode, SafetyMode.NO_NEW_ENTRIES, "-3% -> NO_NEW_ENTRIES")

    # 2% drawdown
    sm3 = SafetyModeManager()
    snap_2 = BrokerSnapshot(account_equity=98000, account_buying_power=40000)
    sm3.evaluate_portfolio_health(snap_2, {}, 100000.0)
    assert_eq(sm3.mode, SafetyMode.LOSS_MITIGATION, "-2% -> LOSS_MITIGATION")

    # Multiple locked -> ORDER_RECONCILIATION_ONLY
    sm4 = SafetyModeManager()
    locked_campaigns = {f"L{i}": SymbolCampaign(symbol=f"L{i}", state="LOCKED_ERROR") for i in range(3)}
    snap_ok = BrokerSnapshot(account_equity=99500, account_buying_power=40000)
    sm4.evaluate_portfolio_health(snap_ok, locked_campaigns, 100000.0)
    assert_eq(sm4.mode, SafetyMode.ORDER_RECONCILIATION_ONLY, "3 locked -> RECONCILIATION_ONLY")


    # ─── Test 13: Target Sizing with Buying Power Cap ────────────────
    print("  [13] Target Sizing with Buying Power Cap...")
    policy_sz = ExecutionPolicy()

    sig_b4 = TradeSignal(symbol="SZ", bucket_id="B4", direction="long", score=0.7,
                         expected_return_bps=10.0, setup_fingerprint="fp",
                         generated_at=datetime.utcnow(), ttl_seconds=90,
                         invalidation_price=99.0)

    # Normal sizing (price=100, invalidation=99, risk=$1/share)
    qty = compute_target_qty(sig_b4, account_equity=100000, price=100.0,
                             atr=1.0, policy=policy_sz, buying_power=50000)
    assert_true(qty > 0, f"target qty > 0 (got {qty})")
    assert_true(qty <= 5000, "respects max_shares_per_symbol")

    # Tiny buying power caps output
    qty_limited = compute_target_qty(sig_b4, account_equity=100000, price=100.0,
                                     atr=1.0, policy=policy_sz, buying_power=500)
    assert_true(qty_limited <= 5.0, f"buying power 500 caps qty (got {qty_limited})")

    # Zero price -> 0
    qty_zero = compute_target_qty(sig_b4, account_equity=100000, price=0.0,
                                  atr=1.0, policy=policy_sz, buying_power=50000)
    assert_eq(qty_zero, 0, "zero price -> zero qty")

    # B2 bucket (scale=0) -> 0
    sig_b2 = TradeSignal(symbol="SZ2", bucket_id="B2", direction="long", score=0.3,
                         expected_return_bps=5.0, setup_fingerprint="fp",
                         generated_at=datetime.utcnow(), ttl_seconds=90)
    qty_b2 = compute_target_qty(sig_b2, account_equity=100000, price=100.0,
                                atr=1.0, policy=policy_sz, buying_power=50000)
    assert_eq(qty_b2, 0, "B2 bucket -> zero qty")

    # ─── Test 14: No-Trade-Zone Gate ─────────────────────────────────
    print("  [14] No-Trade-Zone Gate...")
    policy_ntz = ExecutionPolicy(min_net_edge_bps=3.0)

    sig_good = TradeSignal(symbol="NTZ", bucket_id="B4", direction="long", score=0.8,
                           expected_return_bps=10.0, setup_fingerprint="fp",
                           generated_at=datetime.utcnow(), ttl_seconds=90)
    costs_low = CostEstimate(spread_bps=2.0, slippage_bps=1.0)
    ok, _ = passes_no_trade_zone(sig_good, costs_low, policy_ntz)
    assert_true(ok, "10-3=7bps > 3bps threshold")

    sig_bad = TradeSignal(symbol="NTZ2", bucket_id="B4", direction="long", score=0.8,
                          expected_return_bps=2.0, setup_fingerprint="fp",
                          generated_at=datetime.utcnow(), ttl_seconds=90)
    costs_high = CostEstimate(spread_bps=3.0, slippage_bps=2.0)
    ok, reason = passes_no_trade_zone(sig_bad, costs_high, policy_ntz)
    assert_false(ok, "2-5=-3bps < 3bps -> blocked")
    assert_true("net_edge_too_small" in reason, "reason explains edge")


    # ─── Test 15: Re-Entry Gate ──────────────────────────────────────
    print("  [15] Re-Entry Gate...")
    policy_re = ExecutionPolicy()

    sig_re = TradeSignal(symbol="RE", bucket_id="B4", direction="long", score=0.85,
                         expected_return_bps=10.0, setup_fingerprint="blocked_fp",
                         generated_at=datetime.utcnow(), ttl_seconds=90)

    # Same setup, no reset -> blocked
    c_re = SymbolCampaign(symbol="RE", blocked_setup_fingerprint="blocked_fp")
    ms_no_reset = MarketState(structure_reset=False)
    ok, reason = passes_reentry_gate(sig_re, c_re, ms_no_reset, policy_re)
    assert_false(ok, "same setup no reset -> blocked")

    # Same setup, reset, strong score -> allowed
    ms_reset = MarketState(structure_reset=True)
    ok, _ = passes_reentry_gate(sig_re, c_re, ms_reset, policy_re)
    assert_true(ok, "same setup + reset + strong score -> allowed")

    # Same setup, reset, weak score -> blocked
    sig_weak_re = TradeSignal(symbol="RE", bucket_id="B4", direction="long", score=0.5,
                              expected_return_bps=10.0, setup_fingerprint="blocked_fp",
                              generated_at=datetime.utcnow(), ttl_seconds=90)
    ok, _ = passes_reentry_gate(sig_weak_re, c_re, ms_reset, policy_re)
    assert_false(ok, "same setup + reset + weak score -> blocked")

    # Cooldown active -> blocked
    c_cd = SymbolCampaign(symbol="CD", cooldown_until=datetime.utcnow() + timedelta(seconds=300))
    ok, reason = passes_reentry_gate(sig_re, c_cd, ms_reset, policy_re)
    assert_false(ok, "cooldown active -> blocked")
    assert_true("cooldown_active" in reason, "reason is cooldown")

    # No blocked setup -> allowed
    c_clean = SymbolCampaign(symbol="CLEAN")
    ok, _ = passes_reentry_gate(sig_re, c_clean, ms_no_reset, policy_re)
    assert_true(ok, "no blocked setup -> allowed")

    # ─── Test 16: Contradiction Detection ────────────────────────────
    print("  [16] Contradiction Detection...")

    # Buys open but target=0
    c_contra = SymbolCampaign(symbol="CONTRA", target_qty=0)
    bs_contra = BrokerSymbolState(symbol="CONTRA", position_qty=10, avg_entry_price=100.0,
                                  pending_buy_qty=5, pending_sell_qty=0,
                                  open_buy_order_ids=["ord1"])
    result = resolve_contradictions("CONTRA", c_contra, bs_contra)
    assert_eq(result, "cancel_buys_target_zero", "buys with target=0 detected")

    # Simultaneous buy+sell
    bs_both = BrokerSymbolState(symbol="BOTH", position_qty=10, avg_entry_price=100.0,
                                pending_buy_qty=5, pending_sell_qty=5,
                                open_buy_order_ids=["b1"], open_sell_order_ids=["s1"])
    c_both = SymbolCampaign(symbol="BOTH", target_qty=10)
    result = resolve_contradictions("BOTH", c_both, bs_both)
    assert_eq(result, "lock_conflicting_orders", "simultaneous buy+sell detected")

    # Buy during exit
    c_exit = SymbolCampaign(symbol="EXIT", state="EXITING", target_qty=0)
    bs_exit = BrokerSymbolState(symbol="EXIT", position_qty=5, avg_entry_price=100.0,
                                pending_buy_qty=3, pending_sell_qty=0,
                                open_buy_order_ids=["b2"])
    result = resolve_contradictions("EXIT", c_exit, bs_exit)
    assert_eq(result, "lock_buy_during_exit", "buy during exit detected")

    # Locked with position -> force exit (not unresolved when sell submitted)
    c_locked_pos = SymbolCampaign(symbol="LWP", state="LOCKED_ERROR", target_qty=0,
                                 locked_reason="loss_exceeded_6.0pct")
    bs_locked_pos = BrokerSymbolState(symbol="LWP", position_qty=3, avg_entry_price=50.0,
                                      pending_buy_qty=0, pending_sell_qty=0)
    result = resolve_contradictions("LWP", c_locked_pos, bs_locked_pos)
    assert_eq(result, "locked_with_position", "locked with position detected")

    # Pending sell while locked -> not a contradiction
    bs_locked_exit = BrokerSymbolState(symbol="LWP", position_qty=3, avg_entry_price=50.0,
                                       pending_buy_qty=0, pending_sell_qty=3,
                                       open_sell_order_ids=["s1"])
    result = resolve_contradictions("LWP", c_locked_pos, bs_locked_exit)
    assert_eq(result, None, "locked with pending sell is not a contradiction")

    # No contradiction
    c_ok = SymbolCampaign(symbol="OK", target_qty=10, state="ACTIVE")
    bs_ok = BrokerSymbolState(symbol="OK", position_qty=10, avg_entry_price=100.0,
                              pending_buy_qty=0, pending_sell_qty=0)
    result = resolve_contradictions("OK", c_ok, bs_ok)
    assert_eq(result, None, "no contradiction when aligned")


    # ─── Test 17: Decision Engine ────────────────────────────────────
    print("  [17] Decision Engine...")
    policy_de = ExecutionPolicy()

    # Expired signal -> NO_ACTION
    sig_exp = TradeSignal(symbol="EXP", bucket_id="B4", direction="long", score=0.8,
                          expected_return_bps=10.0, setup_fingerprint="fp",
                          generated_at=datetime.utcnow() - timedelta(seconds=200),
                          ttl_seconds=90)
    c_de = SymbolCampaign(symbol="EXP")
    bs_de = BrokerSymbolState(symbol="EXP", position_qty=0, avg_entry_price=None,
                              pending_buy_qty=0, pending_sell_qty=0)
    ms_de = MarketState(last_price=100.0, atr=1.0)
    costs_de = CostEstimate(spread_bps=2.0, slippage_bps=1.0)
    d = evaluate_signal(sig_exp, c_de, bs_de, ms_de, costs_de, 100000, 100.0, policy_de)
    assert_eq(d.action, "NO_ACTION", "expired signal -> NO_ACTION")
    assert_true("expired" in d.reason, "reason mentions expired")

    # Locked campaign -> NO_ACTION
    c_locked = SymbolCampaign(symbol="LCK", state="LOCKED_ERROR", locked_reason="test")
    sig_lck = TradeSignal(symbol="LCK", bucket_id="B4", direction="long", score=0.8,
                          expected_return_bps=10.0, setup_fingerprint="fp",
                          generated_at=datetime.utcnow(), ttl_seconds=90)
    d = evaluate_signal(sig_lck, c_locked, bs_de, ms_de, costs_de, 100000, 100.0, policy_de)
    assert_eq(d.action, "NO_ACTION", "locked -> NO_ACTION")

    # Flat signal with position -> TARGET_ZERO
    c_active = SymbolCampaign(symbol="ACT", state="ACTIVE", target_qty=50)
    bs_active = BrokerSymbolState(symbol="ACT", position_qty=50, avg_entry_price=100.0,
                                  pending_buy_qty=0, pending_sell_qty=0)
    sig_flat = TradeSignal(symbol="ACT", bucket_id="B4", direction="flat", score=0.8,
                           expected_return_bps=-5.0, setup_fingerprint="fp",
                           generated_at=datetime.utcnow(), ttl_seconds=90)
    d = evaluate_signal(sig_flat, c_active, bs_active, ms_de, costs_de, 100000, 100.0, policy_de)
    assert_eq(d.action, "TARGET_ZERO", "flat signal -> TARGET_ZERO")

    # Good long signal -> SET_TARGET
    sig_long = TradeSignal(symbol="LONG", bucket_id="B4", direction="long", score=0.8,
                           expected_return_bps=10.0, setup_fingerprint="fp_new",
                           generated_at=datetime.utcnow(), ttl_seconds=90,
                           invalidation_price=99.0)
    c_new = SymbolCampaign(symbol="LONG")
    bs_new = BrokerSymbolState(symbol="LONG", position_qty=0, avg_entry_price=None,
                               pending_buy_qty=0, pending_sell_qty=0)
    d = evaluate_signal(sig_long, c_new, bs_new, ms_de, costs_de, 100000, 100.0,
                        policy_de, buying_power=50000)
    assert_eq(d.action, "SET_TARGET", "good signal -> SET_TARGET")
    assert_true(d.target_qty > 0, f"target > 0 (got {d.target_qty})")


    # ─── Test 18: Session Window ─────────────────────────────────────
    print("  [18] Session Window...")
    policy_win = ExecutionPolicy(no_new_entries_before="09:35", no_new_entries_after="15:55")

    class FakeDT:
        def __init__(self, h, m):
            self.h, self.m = h, m
        def strftime(self, fmt):
            return f"{self.h:02d}:{self.m:02d}"

    ok, _ = is_in_trading_window(FakeDT(10, 0), policy_win)
    assert_true(ok, "10:00 is in window")
    ok, _ = is_in_trading_window(FakeDT(9, 30), policy_win)
    assert_false(ok, "09:30 is before window")
    ok, _ = is_in_trading_window(FakeDT(16, 0), policy_win)
    assert_false(ok, "16:00 is after window")

    # ─── Test 19: Campaign Book Sync ─────────────────────────────────
    print("  [19] Campaign Book Sync...")
    cb = CampaignBook(["AAPL", "TSLA"])
    cb.policy_ref = ExecutionPolicy()

    # BUILDING -> ACTIVE when filled
    cb.campaigns["AAPL"].state = "BUILDING"
    cb.campaigns["AAPL"].target_qty = 10
    snap_sync = BrokerSnapshot(account_equity=100000, account_buying_power=50000,
                               positions={"AAPL": BrokerSymbolState(
                                   symbol="AAPL", position_qty=10, avg_entry_price=150.0,
                                   pending_buy_qty=0, pending_sell_qty=0)})
    cb.sync_from_broker(snap_sync)
    assert_eq(cb.campaigns["AAPL"].state, "ACTIVE", "BUILDING->ACTIVE on fill")
    assert_eq(cb.campaigns["AAPL"].entry_price, 150.0, "entry price set from broker")

    # FLAT but broker has position -> adopt
    cb.campaigns["TSLA"].state = "FLAT"
    snap_adopt = BrokerSnapshot(account_equity=100000, account_buying_power=50000,
                                positions={"TSLA": BrokerSymbolState(
                                    symbol="TSLA", position_qty=5, avg_entry_price=200.0,
                                    pending_buy_qty=0, pending_sell_qty=0)})
    cb.sync_from_broker(snap_adopt)
    assert_eq(cb.campaigns["TSLA"].state, "ACTIVE", "FLAT + broker pos -> ACTIVE")
    assert_eq(cb.campaigns["TSLA"].target_qty, 5, "adopts broker position qty")


    # ─── Test 20: Setup Fingerprinting ───────────────────────────────
    print("  [20] Setup Fingerprinting...")
    fp1 = compute_setup_fingerprint("AAPL", "B4", "long", 1.5, "high", "up")
    fp2 = compute_setup_fingerprint("AAPL", "B4", "long", 1.5, "high", "up")
    fp3 = compute_setup_fingerprint("AAPL", "B4", "long", 2.0, "high", "up")
    assert_eq(fp1, fp2, "same inputs -> same fingerprint")
    assert_true(fp1 != fp3, "different z_score -> different fingerprint")
    assert_eq(len(fp1), 12, "fingerprint is 12 chars")

    # ─── Test 21: Broker Snapshot Properties ─────────────────────────
    print("  [21] Broker Snapshot Properties...")
    snap_props = BrokerSnapshot(
        account_equity=100000, account_buying_power=50000,
        positions={
            "A": BrokerSymbolState(symbol="A", position_qty=100, avg_entry_price=50.0,
                                   pending_buy_qty=0, pending_sell_qty=0),
            "B": BrokerSymbolState(symbol="B", position_qty=200, avg_entry_price=25.0,
                                   pending_buy_qty=0, pending_sell_qty=0),
            "C": BrokerSymbolState(symbol="C", position_qty=0, avg_entry_price=None,
                                   pending_buy_qty=10, pending_sell_qty=0),
        })
    assert_eq(snap_props.n_active_positions, 2, "2 active positions (qty>0)")
    assert_eq(snap_props.total_exposure_notional, 100*50.0 + 200*25.0, "notional correct")

    # get_symbol_state for unknown symbol
    bs_unknown = snap_props.get_symbol_state("UNKNOWN")
    assert_eq(bs_unknown.position_qty, 0, "unknown symbol has 0 position")
    assert_eq(bs_unknown.pending_buy_qty, 0, "unknown symbol has 0 pending")

    # ─── Test 22: Effective Exposure ─────────────────────────────────
    print("  [22] Effective Exposure...")
    bs_eff = BrokerSymbolState(symbol="EFF", position_qty=100, avg_entry_price=50.0,
                               pending_buy_qty=20, pending_sell_qty=10)
    assert_eq(bs_eff.effective_exposure, 110, "100+20-10=110")

    bs_eff2 = BrokerSymbolState(symbol="EFF2", position_qty=0, avg_entry_price=None,
                                pending_buy_qty=50, pending_sell_qty=0)
    assert_eq(bs_eff2.effective_exposure, 50, "0+50-0=50")


    # ─── Test 23: net_edge_bps ───────────────────────────────────────
    print("  [23] net_edge_bps...")
    sig_edge = TradeSignal(symbol="E", bucket_id="B4", direction="long", score=0.8,
                           expected_return_bps=15.0, setup_fingerprint="fp",
                           generated_at=datetime.utcnow(), ttl_seconds=90)
    costs_edge = CostEstimate(spread_bps=3.0, slippage_bps=2.0,
                              churn_penalty_bps=1.0, stale_penalty_bps=0.5)
    edge = net_edge_bps(sig_edge, costs_edge)
    assert_eq(edge, 15.0 - 3.0 - 2.0 - 1.0 - 0.5, "net edge = 8.5")

    # ─── Test 24: Campaign Reset ─────────────────────────────────────
    print("  [24] Campaign Reset...")
    c_reset = SymbolCampaign(symbol="RST", state="ACTIVE", target_qty=100,
                             bucket_id="B5", entry_price=150.0, bars_held=20,
                             confirmations=3, best_pnl_pct=0.05)
    old_id = c_reset.campaign_id
    c_reset.reset_flat()
    assert_eq(c_reset.state, "FLAT", "reset -> FLAT")
    assert_eq(c_reset.target_qty, 0, "reset -> target 0")
    assert_eq(c_reset.bucket_id, None, "reset -> no bucket")
    assert_eq(c_reset.entry_price, None, "reset -> no entry price")
    assert_eq(c_reset.confirmations, 0, "reset -> 0 confirmations")
    assert_true(c_reset.campaign_id != old_id, "reset -> new campaign_id")

    # ─── Test 25: Loss Governor Session Lock Persistence ─────────────
    print("  [25] Loss Governor Session Lock Persistence...")
    lg5 = LossGovernor(ExecutionPolicy())

    # First trigger locks
    c_sess = SymbolCampaign(symbol="SESS", state="ACTIVE", entry_price=100.0)
    lg5.evaluate(c_sess, 93.50)  # -6.5% -> lock
    assert_true("SESS" in lg5.session_locked_symbols, "symbol locked in session")

    # Subsequent calls still return lock even if price recovers
    result = lg5.evaluate(c_sess, 100.0)
    assert_eq(result, "lock_session", "session lock persists even on recovery")

    # ─── Test 26: COOLDOWN_BLOCKED State ─────────────────────────────
    print("  [26] COOLDOWN_BLOCKED State...")
    c_cool = SymbolCampaign(symbol="COOL", state="COOLDOWN_BLOCKED")
    d_cool = evaluate_signal(
        TradeSignal(symbol="COOL", bucket_id="B4", direction="long", score=0.9,
                    expected_return_bps=15.0, setup_fingerprint="fp",
                    generated_at=datetime.utcnow(), ttl_seconds=90),
        c_cool,
        BrokerSymbolState(symbol="COOL", position_qty=0, avg_entry_price=None,
                          pending_buy_qty=0, pending_sell_qty=0),
        MarketState(last_price=100.0, atr=1.0),
        CostEstimate(), 100000, 100.0, ExecutionPolicy())
    assert_eq(d_cool.action, "NO_ACTION", "COOLDOWN_BLOCKED -> NO_ACTION")
    assert_true("cooldown" in d_cool.reason, "reason mentions cooldown")


    # ─── Test 27: Profit No-Adding blocks SET_TARGET ─────────────────
    print("  [27] Profit No-Adding blocks SET_TARGET...")
    lg6 = LossGovernor(ExecutionPolicy())
    c_noadd = SymbolCampaign(symbol="NOADD", state="ACTIVE", entry_price=100.0, target_qty=50)
    sig_noadd = TradeSignal(symbol="NOADD", bucket_id="B4", direction="long", score=0.9,
                            expected_return_bps=15.0, setup_fingerprint="fp",
                            generated_at=datetime.utcnow(), ttl_seconds=90,
                            invalidation_price=99.0)
    bs_noadd = BrokerSymbolState(symbol="NOADD", position_qty=50, avg_entry_price=100.0,
                                 pending_buy_qty=0, pending_sell_qty=0)
    ms_noadd = MarketState(last_price=101.0, atr=1.0)  # +1% above entry
    d_noadd = evaluate_signal(sig_noadd, c_noadd, bs_noadd, ms_noadd,
                              CostEstimate(spread_bps=1.0, slippage_bps=1.0),
                              100000, 101.0, ExecutionPolicy(),
                              buying_power=50000, loss_governor=lg6)
    assert_eq(d_noadd.action, "NO_ACTION", "profit no-adding blocks entry")
    assert_true("profit_no_adding" in d_noadd.reason, "reason mentions profit")

    # ─── Test 28: All Forbidden Transitions Are Enforced ─────────────
    print("  [28] All Forbidden Transitions Enforced...")
    for from_state, to_state in FORBIDDEN_TRANSITIONS:
        c_ft = SymbolCampaign(symbol="FT", state=from_state)
        result = c_ft.transition_to(to_state)
        assert_false(result, f"{from_state}->{to_state} forbidden")
        assert_eq(c_ft.state, from_state, f"state unchanged after forbidden {from_state}->{to_state}")

    # ─── Test 29: CampaignBook set_target / set_target_zero ──────────
    print("  [29] CampaignBook set_target / set_target_zero...")
    cb2 = CampaignBook(["X"])
    sig_x = TradeSignal(symbol="X", bucket_id="B5", direction="long", score=0.9,
                        expected_return_bps=15.0, setup_fingerprint="fp_x",
                        generated_at=datetime.utcnow(), ttl_seconds=90,
                        invalidation_price=48.0, category="tech")
    cb2.set_target("X", sig_x, 100, invalidation_price=48.0)
    assert_eq(cb2.campaigns["X"].target_qty, 100, "target set to 100")
    assert_eq(cb2.campaigns["X"].bucket_id, "B5", "bucket set")
    assert_eq(cb2.campaigns["X"].category, "tech", "category set")
    assert_eq(cb2.campaigns["X"].state, "BUILDING", "FLAT->WATCH->BUILDING")
    assert_eq(cb2.campaigns["X"].confirmations, 1, "confirmation counted")

    cb2.set_target_zero("X", "test_exit")
    assert_eq(cb2.campaigns["X"].target_qty, 0, "target zero")
    assert_eq(cb2.campaigns["X"].state, "EXITING", "set_target_zero -> EXITING")
    assert_eq(cb2.campaigns["X"].last_exit_reason, "test_exit", "exit reason stored")
    assert_eq(cb2.campaigns["X"].blocked_setup_fingerprint, "fp_x", "fingerprint blocked")


    # ─── Test 30: Safety Mode Transitions ────────────────────────────
    print("  [30] Safety Mode Transitions...")
    sm5 = SafetyModeManager()

    # Manual set
    sm5.set_mode(SafetyMode.MANUAL_REVIEW_REQUIRED, "test")
    assert_eq(sm5.mode, SafetyMode.MANUAL_REVIEW_REQUIRED, "manual set works")
    assert_false(sm5.allows_new_entries(), "MANUAL blocks entries")
    assert_false(sm5.allows_any_orders(), "MANUAL blocks all orders")

    # De-escalation
    sm6 = SafetyModeManager()
    sm6.set_mode(SafetyMode.NO_NEW_ENTRIES, "dd")
    snap_good = BrokerSnapshot(account_equity=99500, account_buying_power=50000)
    sm6.evaluate_portfolio_health(snap_good, {}, 100000.0)
    assert_eq(sm6.mode, SafetyMode.NORMAL, "de-escalates when conditions improve")

    # ─── Test 31: TradeSignal is_expired ─────────────────────────────
    print("  [31] TradeSignal is_expired...")
    sig_fresh = TradeSignal(symbol="F", bucket_id="B4", direction="long", score=0.8,
                            expected_return_bps=10.0, setup_fingerprint="fp",
                            generated_at=datetime.utcnow(), ttl_seconds=90)
    assert_false(sig_fresh.is_expired, "fresh signal not expired")

    sig_old = TradeSignal(symbol="O", bucket_id="B4", direction="long", score=0.8,
                          expected_return_bps=10.0, setup_fingerprint="fp",
                          generated_at=datetime.utcnow() - timedelta(seconds=100),
                          ttl_seconds=90)
    assert_true(sig_old.is_expired, "old signal is expired")

    # ─── Test 32: Hold Scorer correlation/spread penalty ─────────────
    print("  [32] Hold Scorer spread/correlation penalty...")
    hs2 = HoldScorer()
    c_base = SymbolCampaign(symbol="BASE", state="ACTIVE", bucket_id="B4",
                            entry_price=100.0, bars_held=0)
    score_low_spread = hs2.score(c_base, current_price=100.0, spread_bps=1.0, correlation_penalty=0.0)
    score_high_spread = hs2.score(c_base, current_price=100.0, spread_bps=20.0, correlation_penalty=0.0)
    assert_true(score_low_spread > score_high_spread, "low spread scores higher")

    score_no_corr = hs2.score(c_base, current_price=100.0, spread_bps=3.0, correlation_penalty=0.0)
    score_hi_corr = hs2.score(c_base, current_price=100.0, spread_bps=3.0, correlation_penalty=0.3)
    assert_true(score_no_corr > score_hi_corr, "no correlation scores higher")


    # ─── Test 33: Mock Broker Reconciler ─────────────────────────────
    print("  [33] Mock Broker Reconciler...")

    class MockBroker:
        def __init__(self):
            self.cancelled = []
            self.buys = []
            self.sells = []

        def cancel_order(self, order_id):
            self.cancelled.append(order_id)

        def cancel_all_orders_for_symbol(self, symbol):
            self.cancelled.append(f"ALL_{symbol}")

        def submit_entry_order(self, **kwargs):
            self.buys.append(kwargs)
            return True

        def submit_exit_order(self, **kwargs):
            self.sells.append(kwargs)
            return True

    import pytz
    et = pytz.timezone("US/Eastern")
    mock_broker = MockBroker()
    reconciler = ExecutionReconciler(mock_broker, ExecutionPolicy(), et)

    # Reduce: cancel buys first, then sell
    c_rec = SymbolCampaign(symbol="REC", state="ACTIVE", target_qty=0, campaign_id="rec1")
    bs_rec = BrokerSymbolState(symbol="REC", position_qty=50, avg_entry_price=100.0,
                               pending_buy_qty=10, pending_sell_qty=0,
                               open_buy_order_ids=["buy1", "buy2"])
    result = reconciler.reconcile_symbol("REC", c_rec, bs_rec)
    assert_eq(result, "reducing_exposure", "reduce action returned")
    assert_true("buy1" in mock_broker.cancelled, "buy1 cancelled")
    assert_true("buy2" in mock_broker.cancelled, "buy2 cancelled")
    assert_true(len(mock_broker.sells) == 1, "one sell submitted")
    assert_eq(mock_broker.sells[0]["qty"], 50, "sell full position")

    # In sync
    mock_broker2 = MockBroker()
    reconciler2 = ExecutionReconciler(mock_broker2, ExecutionPolicy(), et)
    c_sync = SymbolCampaign(symbol="SYNC", state="BUILDING", target_qty=100)
    bs_sync = BrokerSymbolState(symbol="SYNC", position_qty=100, avg_entry_price=50.0,
                                pending_buy_qty=0, pending_sell_qty=0)
    result = reconciler2.reconcile_symbol("SYNC", c_sync, bs_sync)
    assert_eq(result, "in_sync", "aligned returns in_sync")
    assert_eq(c_sync.state, "ACTIVE", "BUILDING->ACTIVE on fill sync")

    # ─── Test 34: Reconciler blocks entry for forbidden states ───────
    print("  [34] Reconciler blocks entry for forbidden states...")
    mock_broker3 = MockBroker()
    reconciler3 = ExecutionReconciler(mock_broker3, ExecutionPolicy(), et)

    for blocked_state in ("EXITING", "LOCKED_ERROR", "REDUCING", "COOLDOWN_BLOCKED",
                          "REVALIDATE_LOSER"):
        c_blk = SymbolCampaign(symbol="BLK", state=blocked_state, target_qty=100)
        bs_blk = BrokerSymbolState(symbol="BLK", position_qty=0, avg_entry_price=None,
                                   pending_buy_qty=0, pending_sell_qty=0)
        result = reconciler3._increase_exposure("BLK", c_blk, bs_blk, 100)
        assert_true("entry_blocked" in result, f"{blocked_state} blocks entry")

    # PROTECT_PROFIT blocks adding
    c_pp_blk = SymbolCampaign(symbol="PP", state="PROTECT_PROFIT", target_qty=200)
    bs_pp_blk = BrokerSymbolState(symbol="PP", position_qty=100, avg_entry_price=100.0,
                                  pending_buy_qty=0, pending_sell_qty=0)
    result = reconciler3._increase_exposure("PP", c_pp_blk, bs_pp_blk, 200)
    assert_true("PROTECT_PROFIT" in result, "PROTECT_PROFIT blocks adding")


    # ─── Test 35: Stale Order Cancellation ───────────────────────────
    print("  [35] Stale Order Cancellation...")
    mock_broker4 = MockBroker()
    reconciler4 = ExecutionReconciler(mock_broker4, ExecutionPolicy(), et)

    now = datetime.utcnow()
    stale_snap = BrokerSnapshot(
        account_equity=100000, account_buying_power=50000,
        all_open_orders=[
            {"id": "stale1", "symbol": "AAPL",
             "client_order_id": "BOT|V14_2_1|AAPL|abc|BUY|20240101T120000|deadbe",
             "submitted_at": now - timedelta(seconds=200)},
            {"id": "fresh1", "symbol": "TSLA",
             "client_order_id": "BOT|V14_2_1|TSLA|def|BUY|20240101T120000|aabbcc",
             "submitted_at": now - timedelta(seconds=10)},
            {"id": "manual1", "symbol": "GOOG",
             "client_order_id": "MANUAL_ORDER",
             "submitted_at": now - timedelta(seconds=500)},
        ])

    reconciler4.cancel_stale_orders(stale_snap, now)
    assert_true("stale1" in mock_broker4.cancelled, "stale order cancelled")
    assert_false("fresh1" in mock_broker4.cancelled, "fresh order NOT cancelled")
    assert_false("manual1" in mock_broker4.cancelled, "manual order NOT cancelled")

    # ─── Test 36: Version constant ───────────────────────────────────
    print("  [36] Version constant...")
    assert_eq(VERSION, "V14_2_1", "version is V14_2_1")

    # ─── Test 37: Loss Governor with REVALIDATE_LOSER transition ─────
    print("  [37] REVALIDATE_LOSER transition...")
    lg7 = LossGovernor(ExecutionPolicy())
    c_reval = SymbolCampaign(symbol="REVAL", state="ACTIVE", entry_price=100.0, target_qty=50)
    lg7.apply_loss_action(c_reval, "revalidate")
    assert_eq(c_reval.state, "REVALIDATE_LOSER", "revalidate -> REVALIDATE_LOSER")
    assert_eq(c_reval.target_qty, 50, "revalidate doesn't change target")

    # From REVALIDATE_LOSER, can transition to EXITING
    c_reval2 = SymbolCampaign(symbol="RV2", state="REVALIDATE_LOSER")
    assert_true(c_reval2.transition_to("EXITING"), "REVALIDATE_LOSER->EXITING allowed")
    assert_eq(c_reval2.state, "EXITING", "state changed to EXITING")

    # REVALIDATE_LOSER -> REDUCING allowed
    c_reval3 = SymbolCampaign(symbol="RV3", state="REVALIDATE_LOSER")
    assert_true(c_reval3.transition_to("REDUCING"), "REVALIDATE_LOSER->REDUCING allowed")

    # REVALIDATE_LOSER -> BUILDING forbidden (cannot add size to loser)
    c_reval4 = SymbolCampaign(symbol="RV4", state="REVALIDATE_LOSER")
    assert_false(c_reval4.transition_to("BUILDING"), "REVALIDATE_LOSER->BUILDING FORBIDDEN")
    assert_eq(c_reval4.state, "REVALIDATE_LOSER", "state unchanged after forbidden transition")

    # ─── Test 37b: REVALIDATE_LOSER cannot add size without fresh confirmation ──
    print("  [37b] REVALIDATE_LOSER cannot add size...")
    mock_broker_rv = MockBroker()
    reconciler_rv = ExecutionReconciler(mock_broker_rv, ExecutionPolicy(), et)

    # Scenario: MRNA at -2%, still holding 140 shares, target=140
    # Signal says B4 long but no fresh confirmation
    c_mrna = SymbolCampaign(symbol="MRNA", state="REVALIDATE_LOSER",
                            entry_price=100.0, target_qty=140, campaign_id="mrna1")
    bs_mrna = BrokerSymbolState(symbol="MRNA", position_qty=140, avg_entry_price=100.0,
                                pending_buy_qty=0, pending_sell_qty=0)

    # Attempt to increase exposure (e.g. target bumped to 200)
    result = reconciler_rv._increase_exposure("MRNA", c_mrna, bs_mrna, 200)
    assert_true("entry_blocked" in result, f"REVALIDATE_LOSER blocks increase: {result}")
    assert_true("REVALIDATE_LOSER" in result, "reason mentions REVALIDATE_LOSER")
    assert_eq(len(mock_broker_rv.buys), 0, "no buy order submitted for loser")

    # Verify reconcile_symbol also blocks (target > effective)
    c_mrna2 = SymbolCampaign(symbol="MRNA2", state="REVALIDATE_LOSER",
                             entry_price=100.0, target_qty=200, campaign_id="mrna2")
    bs_mrna2 = BrokerSymbolState(symbol="MRNA2", position_qty=140, avg_entry_price=100.0,
                                 pending_buy_qty=0, pending_sell_qty=0)
    result2 = reconciler_rv.reconcile_symbol("MRNA2", c_mrna2, bs_mrna2)
    assert_true("entry_blocked" in result2, f"reconcile_symbol blocks REVALIDATE_LOSER add: {result2}")
    assert_eq(len(mock_broker_rv.buys), 0, "still no buys submitted")


    # ─── Test 38: PortfolioAdmission record_admission ────────────────
    print("  [38] PortfolioAdmission record_admission...")
    adm3 = PortfolioAdmission(ExecutionPolicy())
    adm3.reset_day("2024-01-01")
    adm3.reset_cycle()
    assert_eq(adm3.entries_this_cycle, 0, "cycle starts at 0")
    assert_eq(adm3.entries_today, 0, "day starts at 0")
    adm3.record_admission()
    assert_eq(adm3.entries_this_cycle, 1, "cycle incremented")
    assert_eq(adm3.entries_today, 1, "day incremented")
    adm3.record_admission()
    assert_eq(adm3.entries_this_cycle, 2, "cycle=2")
    assert_eq(adm3.entries_today, 2, "day=2")

    # ─── Test 39: Reconciler resolve_all_contradictions ──────────────
    print("  [39] Reconciler resolve_all_contradictions...")
    mock_broker5 = MockBroker()
    reconciler5 = ExecutionReconciler(mock_broker5, ExecutionPolicy(), et)

    campaigns_contra = {
        "A": SymbolCampaign(symbol="A", target_qty=0, state="ACTIVE"),
        "B": SymbolCampaign(symbol="B", target_qty=50, state="ACTIVE"),
    }
    snap_contra = BrokerSnapshot(
        account_equity=100000, account_buying_power=50000,
        positions={
            "A": BrokerSymbolState(symbol="A", position_qty=10, avg_entry_price=100.0,
                                   pending_buy_qty=5, pending_sell_qty=0,
                                   open_buy_order_ids=["a_buy1"]),
            "B": BrokerSymbolState(symbol="B", position_qty=50, avg_entry_price=100.0,
                                   pending_buy_qty=0, pending_sell_qty=0),
        })
    count = reconciler5.resolve_all_contradictions(campaigns_contra, snap_contra)
    assert_eq(count, 1, "1 contradiction resolved (A has buys with target=0)")
    assert_true("a_buy1" in mock_broker5.cancelled, "A's buy cancelled")
    assert_eq(campaigns_contra["A"].state, "EXITING", "A moved to EXITING")

    # ─── Test 40: Full Integration (decision->campaign->reconcile) ───
    print("  [40] Full Integration Flow...")
    mock_broker6 = MockBroker()
    reconciler6 = ExecutionReconciler(mock_broker6, ExecutionPolicy(), et)
    cb3 = CampaignBook(["INTG"])
    cb3.policy_ref = ExecutionPolicy()
    policy_intg = ExecutionPolicy()
    lg8 = LossGovernor(policy_intg)

    # Fresh signal for new symbol
    sig_intg = TradeSignal(symbol="INTG", bucket_id="B4", direction="long", score=0.85,
                           expected_return_bps=12.0, setup_fingerprint="fp_intg",
                           generated_at=datetime.utcnow(), ttl_seconds=90,
                           invalidation_price=98.0, category="general")

    snap_intg = BrokerSnapshot(account_equity=100000, account_buying_power=50000,
                               positions={})
    bs_intg = snap_intg.get_symbol_state("INTG")
    ms_intg = MarketState(last_price=100.0, atr=1.5)
    costs_intg = CostEstimate(spread_bps=2.0, slippage_bps=1.0)

    d_intg = evaluate_signal(sig_intg, cb3.get("INTG"), bs_intg, ms_intg,
                             costs_intg, 100000, 100.0, policy_intg,
                             buying_power=50000, loss_governor=lg8)
    assert_eq(d_intg.action, "SET_TARGET", "integration: SET_TARGET")
    assert_true(d_intg.target_qty > 0, "integration: positive target")

    # Apply to campaign book
    cb3.set_target("INTG", sig_intg, d_intg.target_qty, sig_intg.invalidation_price)
    assert_eq(cb3.campaigns["INTG"].state, "BUILDING", "integration: BUILDING after set_target")

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    if errors:
        for e in errors:
            print(e)
    print()
    return failed == 0



# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    success = _run_regression_tests()
    import sys
    sys.exit(0 if success else 1)
