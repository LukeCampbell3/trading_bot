"""Directional hysteresis for option entries.

The guard is intentionally stateful.  A route score changing from CALL to PUT
for one minute is not itself a reversal signal.  Initial direction is admitted
with short consensus.  Once a bias is established, the opposite side must be
both stronger and persistent before the bias can flip.  Open/pending option
risk blocks opposite-side entries entirely.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class GuardDecision:
    allowed_candidates: list = field(default_factory=list)
    bias: str = "NEUTRAL"
    proposed_side: str = "NEUTRAL"
    reason: str = ""
    top_call_score: float = 0.0
    top_put_score: float = 0.0
    score_edge: float = 0.0
    consensus_count: int = 0
    flip_streak: int = 0
    size_multiplier: float = 1.0
    bias_changed: bool = False


@dataclass
class DirectionState:
    bias: str = "NEUTRAL"
    side_history: deque = field(default_factory=lambda: deque(maxlen=4))
    bars_in_bias: int = 0
    flip_candidate: str = "NEUTRAL"
    flip_streak: int = 0
    cooldown_remaining: int = 0
    last_exit_side: str = "NEUTRAL"
    last_exit_reason: str = ""
    last_decision_reason: str = ""


class OptionsReversalGuard:
    """Prevent unstable CALL/PUT oscillation without suppressing real reversals."""

    def __init__(self, config: dict):
        self.cfg = config
        self._states: Dict[str, DirectionState] = {}

    def state(self, symbol: str) -> DirectionState:
        symbol = symbol.upper()
        st = self._states.get(symbol)
        if st is None:
            st = DirectionState(
                side_history=deque(maxlen=int(self.cfg.get("direction_consensus_window", 4)))
            )
            self._states[symbol] = st
        return st

    @staticmethod
    def _top_by_side(candidates: Iterable) -> Tuple[Optional[object], Optional[object]]:
        call = None
        put = None
        for c in candidates:
            side = str(getattr(c, "side", "")).upper()
            if side == "CALL" and (call is None or c.score > call.score):
                call = c
            elif side == "PUT" and (put is None or c.score > put.score):
                put = c
        return call, put

    def select_candidates(
        self,
        symbol: str,
        candidates: List,
        *,
        price: float,
        vwap: float,
        atr: float,
        open_risk_side: str = "NEUTRAL",
        has_pending_risk: bool = False,
    ) -> GuardDecision:
        """Return candidates allowed by the current direction state.

        `open_risk_side` should be CALL/PUT when a filled or confirmed strategy
        already has directional exposure.  `has_pending_risk` includes submitted
        entries.  Opposite entries are not allowed while that risk exists.
        """
        st = self.state(symbol)
        call, put = self._top_by_side(candidates)
        call_score = float(call.score) if call is not None else 0.0
        put_score = float(put.score) if put is not None else 0.0
        proposed = "CALL" if call_score >= put_score and call_score > 0 else (
            "PUT" if put_score > 0 else "NEUTRAL"
        )
        best_score = max(call_score, put_score)
        other_score = put_score if proposed == "CALL" else call_score
        edge = best_score - other_score if proposed != "NEUTRAL" else 0.0

        if st.cooldown_remaining > 0:
            st.cooldown_remaining -= 1

        # Record only a meaningful directional observation.  Noise below the
        # watch threshold should not manufacture consensus.
        min_bias = float(self.cfg.get("direction_min_bias_score", 0.58))
        observed = proposed if best_score >= min_bias else "NEUTRAL"
        st.side_history.append(observed)

        decision = GuardDecision(
            bias=st.bias,
            proposed_side=proposed,
            top_call_score=call_score,
            top_put_score=put_score,
            score_edge=edge,
            flip_streak=st.flip_streak,
        )

        # Existing broker risk owns the directional lock.  The guard may continue
        # to learn the opposite signal, but it cannot create hedging-by-accident.
        risk_side = str(open_risk_side or "NEUTRAL").upper()
        if risk_side in ("CALL", "PUT"):
            if st.bias == "NEUTRAL":
                st.bias = risk_side
                st.bars_in_bias = 1
            elif st.bias != risk_side:
                st.bias = risk_side
                st.bars_in_bias = 1
            same = [c for c in candidates if str(c.side).upper() == risk_side]
            st.bars_in_bias += 1
            st.flip_candidate = "NEUTRAL"
            st.flip_streak = 0
            decision.allowed_candidates = same
            decision.bias = st.bias
            decision.reason = "open_risk_direction_lock"
            decision.size_multiplier = 1.0
            st.last_decision_reason = decision.reason
            return decision

        if has_pending_risk and self.cfg.get("block_opposite_while_pending", True):
            # Pending entries are treated as a temporary directional lock if the
            # guard already has a bias.  Do not create a contradictory order.
            if st.bias in ("CALL", "PUT"):
                decision.allowed_candidates = [
                    c for c in candidates if str(c.side).upper() == st.bias
                ]
                decision.reason = "pending_order_direction_lock"
                decision.bias = st.bias
                decision.size_multiplier = self._size_multiplier(st)
                st.last_decision_reason = decision.reason
                return decision

        # Establish initial bias with short consensus and an edge over the other
        # side.  This normally costs only one extra observation, not a long warm-up.
        if st.bias == "NEUTRAL":
            min_edge = float(self.cfg.get("direction_min_score_edge", 0.06))
            required = int(self.cfg.get("direction_consensus_required", 2))
            consensus = sum(1 for x in st.side_history if x == proposed)
            decision.consensus_count = consensus
            if proposed == "NEUTRAL" or best_score < min_bias:
                decision.reason = "no_direction_above_floor"
                return decision
            if other_score > 0 and edge < min_edge:
                decision.reason = "direction_conflict_initial"
                return decision
            if consensus < required:
                decision.reason = "direction_consensus_building"
                return decision
            st.bias = proposed
            st.bars_in_bias = 1
            st.flip_candidate = "NEUTRAL"
            st.flip_streak = 0
            decision.bias = st.bias
            decision.bias_changed = True
            decision.allowed_candidates = [c for c in candidates if str(c.side).upper() == st.bias]
            decision.reason = "initial_bias_locked"
            decision.size_multiplier = self._size_multiplier(st)
            st.last_decision_reason = decision.reason
            return decision

        # Hysteresis: while the existing side is still present, keep it unless the
        # opposite side clears the much stronger reversal gate.
        same_score = call_score if st.bias == "CALL" else put_score
        opposite = "PUT" if st.bias == "CALL" else "CALL"
        opposite_score = put_score if opposite == "PUT" else call_score
        opposite_edge = opposite_score - same_score

        reversal_min = float(self.cfg.get("reversal_min_route_score", 0.68))
        reversal_edge = float(self.cfg.get("reversal_min_score_edge", 0.12))
        min_vwap_atr = float(self.cfg.get("reversal_min_vwap_distance_atr", 0.10))
        vwap_distance_atr = abs(price - vwap) / atr if atr and atr > 0 else 0.0
        vwap_supports_opposite = (
            (opposite == "CALL" and price > vwap) or
            (opposite == "PUT" and price < vwap)
        ) and vwap_distance_atr >= min_vwap_atr

        reversal_eligible = (
            opposite_score >= reversal_min
            and opposite_edge >= reversal_edge
            and vwap_supports_opposite
            and st.cooldown_remaining <= 0
        )

        if reversal_eligible:
            if st.flip_candidate == opposite:
                st.flip_streak += 1
            else:
                st.flip_candidate = opposite
                st.flip_streak = 1
            decision.flip_streak = st.flip_streak
            required_flip = int(self.cfg.get("reversal_consecutive_bars", 3))
            if st.flip_streak >= required_flip:
                old = st.bias
                st.bias = opposite
                st.bars_in_bias = 1
                st.flip_candidate = "NEUTRAL"
                st.flip_streak = 0
                st.cooldown_remaining = int(self.cfg.get("reversal_cooldown_bars", 4))
                decision.bias = st.bias
                decision.bias_changed = True
                decision.allowed_candidates = [c for c in candidates if str(c.side).upper() == st.bias]
                decision.reason = f"confirmed_reversal:{old}->{st.bias}"
                decision.size_multiplier = float(self.cfg.get("recent_reversal_size_multiplier", 0.50))
                st.last_decision_reason = decision.reason
                return decision
            decision.allowed_candidates = [c for c in candidates if str(c.side).upper() == st.bias]
            decision.reason = "reversal_confirmation_building"
            decision.size_multiplier = self._size_multiplier(st)
            st.bars_in_bias += 1
            st.last_decision_reason = decision.reason
            return decision

        # Opposite impulse failed the reversal standard; clear the streak and keep
        # the established side.  This is the main anti-whipsaw behavior.
        st.flip_candidate = "NEUTRAL"
        st.flip_streak = 0
        st.bars_in_bias += 1
        decision.allowed_candidates = [c for c in candidates if str(c.side).upper() == st.bias]
        decision.reason = "bias_hysteresis_hold"
        decision.size_multiplier = self._size_multiplier(st)
        st.last_decision_reason = decision.reason
        return decision

    def ticket_side_allowed(self, symbol: str, side: str) -> Tuple[bool, str]:
        st = self.state(symbol)
        side = str(side).upper().replace("TICKETSIDE.", "")
        if st.bias == "NEUTRAL":
            return False, "no_locked_direction_bias"
        if side != st.bias:
            return False, f"ticket_side_conflicts_with_bias:{side}!={st.bias}"
        return True, "ok"

    def record_exit(self, symbol: str, side: str, reason: str) -> None:
        st = self.state(symbol)
        side = str(side).upper().replace("TICKETSIDE.", "")
        st.last_exit_side = side
        st.last_exit_reason = str(reason or "")
        reason_l = st.last_exit_reason.lower()
        if "stop" in reason_l or "loss" in reason_l:
            st.cooldown_remaining = max(
                st.cooldown_remaining,
                int(self.cfg.get("stop_reversal_cooldown_bars", 7)),
            )
        else:
            st.cooldown_remaining = max(
                st.cooldown_remaining,
                int(self.cfg.get("target_reentry_cooldown_bars", 2)),
            )

    def force_bias(self, symbol: str, side: str) -> None:
        """Used only to reconcile direction state with broker truth."""
        st = self.state(symbol)
        side = str(side).upper()
        if side not in ("CALL", "PUT", "NEUTRAL"):
            raise ValueError(f"invalid side: {side}")
        st.bias = side
        st.bars_in_bias = 1 if side != "NEUTRAL" else 0
        st.flip_candidate = "NEUTRAL"
        st.flip_streak = 0

    def _size_multiplier(self, st: DirectionState) -> float:
        if st.cooldown_remaining > 0 and st.last_exit_side != "NEUTRAL":
            return float(self.cfg.get("recent_reversal_size_multiplier", 0.50))
        full_after = int(self.cfg.get("stable_bias_full_size_after_bars", 4))
        if st.bars_in_bias < full_after:
            return float(self.cfg.get("new_bias_size_multiplier", 0.75))
        return 1.0
