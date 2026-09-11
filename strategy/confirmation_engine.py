"""
Confirmation Engine for V14.2

A watch ticket may be confirmed ONLY if ALL conditions are true:
- Underlying moved at least confirm_min_directional_atr in expected direction
- Adverse move before confirmation did not exceed confirm_max_adverse_atr
- MFE velocity is above confirm_min_mfe_velocity
- Price remains on the correct side of VWAP for the route
- Route has not decayed
- Environment stress remains acceptable
- Option spread is still tradable at next quote after confirmation

CRITICAL: Do NOT use the favorable quote from the confirmation moment.
Use the NEXT available option quote after confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG
from strategy.watch_ticket import WatchTicket, TicketSide, TicketStatus


@dataclass
class ConfirmationResult:
    """Result of a confirmation check."""
    confirmed: bool = False
    reason: str = ""
    directional_move_atr: float = 0.0
    adverse_move_atr: float = 0.0
    mfe_velocity: float = 0.0
    current_price: float = 0.0
    vwap_status: str = ""
    env_stress: float = 0.0


class ConfirmationEngine:
    """
    Evaluates whether a WATCHING ticket should transition to CONFIRMED.
    
    For CALL routes: directional movement is upward. VWAP confirmation means
    price remains above VWAP or successfully reclaims VWAP depending on route.
    
    For PUT routes: directional movement is downward. VWAP confirmation means
    price remains below VWAP or rejects VWAP depending on route.
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or CFG

    def check_confirmation(
        self,
        ticket: WatchTicket,
        current_price: float,
        current_vwap: float,
        current_atr: float,
        high_since_watch: float,
        low_since_watch: float,
        mfe_velocity: float,
        env_stress: float,
        route_score_now: float,
        option_quote_valid: bool = True,
        timestamp: Optional[datetime] = None,
        cfg_override: Optional[dict] = None,
    ) -> ConfirmationResult:
        """
        Check all confirmation conditions for a watching ticket.

        Parameters:
            ticket: The watch ticket to evaluate
            current_price: Current underlying price
            current_vwap: Current VWAP value
            current_atr: Current ATR value (for normalizing moves)
            high_since_watch: Highest price since watch creation
            low_since_watch: Lowest price since watch creation
            mfe_velocity: Rate of favorable movement (normalized)
            env_stress: Current environment stress level [0,1]
            route_score_now: Current route score (decay check)
            option_quote_valid: Whether next option quote is tradable
            timestamp: Optional timestamp for the check
            cfg_override: optional per-route confidence-adjusted thresholds
                (see strategy.dynamic_gate_controller); missing keys fall
                back to the base config this engine was constructed with.

        Returns:
            ConfirmationResult with pass/fail and diagnostics
        """
        cfg = {**self.cfg, **(cfg_override or {})}
        result = ConfirmationResult()
        result.current_price = current_price
        result.mfe_velocity = mfe_velocity
        result.env_stress = env_stress

        if ticket.status != TicketStatus.WATCHING:
            result.reason = "ticket_not_watching"
            return result

        if current_atr <= 0:
            result.reason = "invalid_atr"
            return result

        # ─── Directional Move ────────────────────────────────────────────
        if ticket.side == TicketSide.CALL:
            directional_move = current_price - ticket.underlying_price_at_watch
            adverse_move = ticket.underlying_price_at_watch - low_since_watch
        else:  # PUT
            directional_move = ticket.underlying_price_at_watch - current_price
            adverse_move = high_since_watch - ticket.underlying_price_at_watch

        directional_move_atr = directional_move / current_atr if current_atr > 0 else 0.0
        adverse_move_atr = adverse_move / current_atr if current_atr > 0 else 0.0

        result.directional_move_atr = directional_move_atr
        result.adverse_move_atr = adverse_move_atr

        # Check 1: Minimum directional move
        if directional_move_atr < cfg["confirm_min_directional_atr"]:
            result.reason = (
                f"insufficient_directional_move: {directional_move_atr:.4f} "
                f"< {cfg['confirm_min_directional_atr']}"
            )
            return result

        # Check 2: Adverse move limit
        if adverse_move_atr > cfg["confirm_max_adverse_atr"]:
            result.reason = (
                f"excessive_adverse_move: {adverse_move_atr:.4f} "
                f"> {cfg['confirm_max_adverse_atr']}"
            )
            return result

        # Check 3: MFE velocity
        if mfe_velocity < cfg["confirm_min_mfe_velocity"]:
            result.reason = (
                f"insufficient_mfe_velocity: {mfe_velocity:.4f} "
                f"< {cfg['confirm_min_mfe_velocity']}"
            )
            return result

        # Check 4: VWAP positioning
        vwap_ok, vwap_status = self._check_vwap(ticket, current_price, current_vwap)
        result.vwap_status = vwap_status
        if not vwap_ok:
            result.reason = f"vwap_failure: {vwap_status}"
            return result

        # Check 5: Route decay
        if route_score_now < cfg["watch_min_route_score"] * 0.9:
            result.reason = f"route_decayed: {route_score_now:.4f}"
            return result

        # Check 6: Environment stress
        if env_stress > cfg["max_env_stress"]:
            result.reason = f"env_stress_exceeded: {env_stress:.4f} > {cfg['max_env_stress']}"
            return result

        # Check 7: Option quote tradability (NEXT quote, not confirmation moment)
        if not option_quote_valid:
            result.reason = "option_quote_invalid_or_stale"
            return result

        # All checks passed
        result.confirmed = True
        result.reason = "confirmed"
        return result

    def _check_vwap(
        self, ticket: WatchTicket, price: float, vwap: float
    ) -> Tuple[bool, str]:
        """
        Check VWAP positioning based on side and route.
        
        For CALL routes: price should be above VWAP or reclaiming.
        For PUT routes: price should be below VWAP or rejecting.
        """
        if ticket.side == TicketSide.CALL:
            if ticket.route == "VWAP_PULLBACK":
                # Price pulled back to VWAP and is now reclaiming
                if price >= vwap * 0.998:  # allow 0.2% tolerance for reclaim
                    return True, "call_reclaim_ok"
                return False, "call_below_vwap_pullback"
            elif ticket.route in ("PULLBACK_CONTINUATION", "DOMINANT_TREND_PULLBACK"):
                # Price should be above VWAP
                if price >= vwap:
                    return True, "call_above_vwap"
                return False, "call_below_vwap"
            else:
                # General: price above VWAP
                if price >= vwap * 0.995:
                    return True, "call_near_vwap_ok"
                return False, "call_too_far_below_vwap"

        else:  # PUT
            if ticket.route == "PUT_REJECTION":
                # Price rejected from VWAP area and is below
                if price <= vwap * 1.002:
                    return True, "put_rejection_ok"
                return False, "put_above_vwap_rejection"
            else:
                # General put: price below VWAP
                if price <= vwap:
                    return True, "put_below_vwap"
                return False, "put_above_vwap"

    def apply_confirmation(
        self,
        ticket: WatchTicket,
        result: ConfirmationResult,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        If confirmation result is positive, transition the ticket.
        Returns True if ticket was confirmed.
        """
        if not result.confirmed:
            return False

        ticket.mark_confirmed(
            directional_move_atr=result.directional_move_atr,
            adverse_move_atr=result.adverse_move_atr,
            mfe_velocity=result.mfe_velocity,
            price=result.current_price,
            timestamp=timestamp,
        )
        return True
