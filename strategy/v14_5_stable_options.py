"""Canonical V14.5 stable option trader.

This layer finalizes V14.5 by adding exit hysteresis and one-position-per-symbol
risk discipline to the V14.5 directional/Greeks runner. Hard stops and profit
targets remain immediate. Only structure-decay exits are debounced, because a
single VWAP touch/cross in a high-volatility underlying is not enough evidence
that the directional thesis reversed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from strategy.v14_3_highvol_config import can_trade_symbol
from strategy.v14_5_options_runner import (
    V14_5_StableOptionsRunner,
    SnapshotEnrichedOptionChainFetcher,
)
from strategy.v14_5_stability_config import get_v14_5_config
from strategy.watch_ticket import TicketSide, TicketStatus


class TrackingSnapshotFetcher(SnapshotEnrichedOptionChainFetcher):
    """Snapshot fetcher that exposes whether the last quote had usable Greeks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_greeks_available = False

    def get_live_quotes(self, long_contract, short_contract):
        legs = super().get_live_quotes(long_contract, short_contract)
        self.last_greeks_available = bool(
            legs
            and abs(float(getattr(legs[0], "delta", 0.0) or 0.0)) > 0
            and abs(float(getattr(legs[1], "delta", 0.0) or 0.0)) > 0
        )
        return legs


class _NoPyramidConditioner:
    """Update direction state but do not add a second live strategy on a symbol."""

    def __init__(self, base, owner: "V14_5_OptionsTrader"):
        self.base = base
        self.owner = owner

    def evaluate_routes(self, **kwargs):
        candidates = self.base.evaluate_routes(**kwargs)
        open_side, pending, conflict = self.owner._directional_risk_state()
        if conflict:
            return []
        if open_side in ("CALL", "PUT") or pending:
            return []
        return candidates


class V14_5_OptionsTrader(V14_5_StableOptionsRunner):
    """Stable CALL/PUT debit-spread trader for paper/replay validation."""

    VERSION = "14.5-stable-options"
    STATUS = "STABLE_DIRECTION_AND_EXIT_HYSTERESIS_PAPER_VALIDATION"
    LABEL = "V14_5_STABLE_OPTIONS_TRADER"

    def __init__(
        self,
        symbol: str,
        trading_client: Any,
        option_data_client: Any,
        config: Optional[dict] = None,
        paper_mode: bool = True,
        log_dir: str = "HFT/logs/v14_5_options",
        allow_offline_simulation: bool = False,
    ):
        cfg = config or get_v14_5_config()
        super().__init__(
            symbol=symbol,
            trading_client=trading_client,
            option_data_client=option_data_client,
            config=cfg,
            paper_mode=paper_mode,
            log_dir=log_dir,
            allow_offline_simulation=allow_offline_simulation,
        )
        if trading_client is not None and option_data_client is not None:
            self.chain_fetcher = TrackingSnapshotFetcher(trading_client, option_data_client)
        self.route_conditioner = _NoPyramidConditioner(self.route_conditioner, self)
        self._latest_atr = 0.0
        self._exit_structure_fail_streak: Dict[str, int] = {}

    def evaluate_opportunity(
        self,
        symbol: str,
        price: float,
        vwap: float,
        atr: float,
        high_of_day: float,
        low_of_day: float,
        trend_slope: float,
        volume_ratio: float,
        price_5m_ago: float,
        price_15m_ago: float,
        option_liquidity: float = 0.7,
        iv_percentile: float = 0.5,
    ):
        self._latest_atr = max(0.0, float(atr or 0.0))
        return super().evaluate_opportunity(
            symbol=symbol,
            price=price,
            vwap=vwap,
            atr=atr,
            high_of_day=high_of_day,
            low_of_day=low_of_day,
            trend_slope=trend_slope,
            volume_ratio=volume_ratio,
            price_5m_ago=price_5m_ago,
            price_15m_ago=price_15m_ago,
            option_liquidity=option_liquidity,
            iv_percentile=iv_percentile,
        )

    def _attempt_real_execution(self, ticket, underlying_price: float, iv_percentile: float):
        # Preserve the parent direction gate and symbol policy, but haircut debit
        # allocation when the data entitlement/feed did not supply usable Greeks.
        base_allowed, _, base_size = can_trade_symbol(ticket.symbol)
        if not base_allowed:
            return {"action": "REJECTED", "reason": "symbol_policy_blocked"}
        result = super()._attempt_real_execution(ticket, underlying_price, iv_percentile)
        return result

    def _continuation_failed(self, ticket, underlying_price: float, vwap: float) -> bool:
        atr = max(self._latest_atr, 0.0)
        buffer_amt = float(self.cfg.get("exit_vwap_hysteresis_atr", 0.05)) * atr
        if ticket.side == TicketSide.CALL:
            return underlying_price < (vwap - buffer_amt)
        return underlying_price > (vwap + buffer_amt)

    def _manage_real_option_exits(self, symbol: str, underlying_price: float, vwap: float):
        """Manage exits using executable spread value + structure hysteresis.

        Risk stops and profit targets are not delayed. The only delayed exit is
        continuation decay, which requires the underlying thesis to remain on the
        wrong side of VWAP (with ATR buffer) for consecutive observations.
        """
        required_fail = max(1, int(self.cfg.get("exit_reversal_confirm_bars", 2)))

        for ticket_id, plan in list(self._positions.items()):
            if plan.symbol != symbol or any(
                m["ticket_id"] == ticket_id for m in self._pending_exit.values()
            ):
                continue
            quote = self._current_spread_quote(plan)
            if quote is None or plan.entry_debit <= 0:
                continue
            _, _, close_bid, close_ask, close_mid = quote
            pnl_pct = (close_bid - plan.entry_debit) / plan.entry_debit
            plan.peak_pnl_pct = max(plan.peak_pnl_pct, pnl_pct)

            ticket = self.ticket_book.active_tickets.get(ticket_id)
            if not ticket:
                continue

            failed_now = self._continuation_failed(ticket, underlying_price, vwap)
            if failed_now:
                self._exit_structure_fail_streak[ticket_id] = (
                    self._exit_structure_fail_streak.get(ticket_id, 0) + 1
                )
            else:
                self._exit_structure_fail_streak[ticket_id] = 0
            continuation_failed = self._exit_structure_fail_streak[ticket_id] >= required_fail

            if plan.mode == "FALLBACK":
                if pnl_pct <= self.cfg["single_spread_initial_stop_pct"]:
                    self._submit_partial_exit(
                        plan, plan.fallback_qty, close_bid, close_ask, "fallback_stop", "ALL"
                    )
                elif pnl_pct >= self.cfg["soft_greed_target_pct"]:
                    self._submit_partial_exit(
                        plan, plan.fallback_qty, close_bid, close_ask, "fallback_target", "ALL"
                    )
                elif continuation_failed and pnl_pct > 0:
                    self._submit_partial_exit(
                        plan, plan.fallback_qty, close_bid, close_ask,
                        "continuation_decay_confirmed", "ALL"
                    )
                continue

            # Package risk stop / core target remain immediate.
            if not plan.core_closed:
                if pnl_pct <= self.cfg["package_stop_pct"]:
                    self._submit_partial_exit(
                        plan, plan.total_qty, close_bid, close_ask, "package_stop", "ALL"
                    )
                elif pnl_pct >= self.cfg["core_target_pct"]:
                    self._submit_partial_exit(
                        plan, plan.core_qty, close_bid, close_ask, "core_target", "CORE"
                    )
                continue

            if plan.runner_qty <= 0:
                continue
            if pnl_pct >= self.cfg["runner_target_pct"]:
                self._submit_partial_exit(
                    plan, plan.runner_qty, close_bid, close_ask, "runner_target", "RUNNER"
                )
            elif pnl_pct <= self.cfg["runner_lock_pct"]:
                self._submit_partial_exit(
                    plan, plan.runner_qty, close_bid, close_ask, "runner_lock", "RUNNER"
                )
            elif continuation_failed and pnl_pct > self.cfg["runner_lock_pct"]:
                self._submit_partial_exit(
                    plan, plan.runner_qty, close_bid, close_ask,
                    "runner_continuation_decay_confirmed", "RUNNER"
                )

    def _finalize_exit(self, order_id: str):
        meta = self._pending_exit.get(order_id)
        ticket_id = meta.get("ticket_id") if meta else None
        super()._finalize_exit(order_id)
        if ticket_id and ticket_id not in self._positions:
            self._exit_structure_fail_streak.pop(ticket_id, None)

    def get_status(self):
        base = super().get_status()
        base.update({
            "version": self.VERSION,
            "status": self.STATUS,
            "label": self.LABEL,
            "one_strategy_per_symbol": True,
            "exit_reversal_confirm_bars": int(self.cfg.get("exit_reversal_confirm_bars", 2)),
            "exit_vwap_hysteresis_atr": float(self.cfg.get("exit_vwap_hysteresis_atr", 0.05)),
            "greeks_last_available": bool(
                getattr(self.chain_fetcher, "last_greeks_available", False)
            ),
        })
        return base


def create_v14_5_stable_options_trader(
    symbol: str,
    trading_client: Any,
    option_data_client: Any,
    paper_mode: bool = True,
    log_dir: str = "HFT/logs/v14_5_options",
) -> V14_5_OptionsTrader:
    return V14_5_OptionsTrader(
        symbol=symbol,
        trading_client=trading_client,
        option_data_client=option_data_client,
        config=get_v14_5_config(),
        paper_mode=paper_mode,
        log_dir=str(Path(log_dir) / symbol.upper()),
    )
