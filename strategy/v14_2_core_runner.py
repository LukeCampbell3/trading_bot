"""
V14.2 Core Runner Strategy
===========================
Status: SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN

Replaces V12.2 as the active strategy candidate inside robust stressed synthetic simulation.
NOT live-proven. Runs in paper/simulation mode by default.
Live trading DISABLED unless explicitly enabled through guarded config flag.

Core Doctrine:
- Base signal is WATCH-ONLY. Never commit capital at the noisy base-signal stage.
- Capital enters ONLY after confirmation + spread-quality validation.
- CALL debit spreads and PUT debit spreads ONLY.
- Limit-only multileg orders.
- Bot-managed exits (not native multileg stops).
- Prefer core-runner package when conditions qualify.
- V12.2 single-spread fallback when package conditions do not qualify.
- Log every skipped, watched, confirmed, filled, missed, and exited opportunity.
"""

from __future__ import annotations

import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG, get_config
from strategy.watch_ticket import (
    WatchTicket, WatchTicketBook, TicketSide, TicketStatus
)
from strategy.confirmation_engine import ConfirmationEngine, ConfirmationResult
from strategy.spread_quality_gate import SpreadQualityGate, OptionLeg, SpreadQualityReport
from strategy.package_builder import PackageBuilder, PackageResult
from strategy.route_conditioning import RouteConditioner, RouteCandidate
from strategy.risk_manager import RiskManager

from execution.mleg_execution_manager import (
    MlegExecutionManager, MlegOrder, MlegLeg, OrderState
)
from execution.fill_quality_logger import FillQualityLogger

from telemetry.route_expectancy_monitor import RouteExpectancyMonitor
from telemetry.environment_expectancy_monitor import (
    EnvironmentExpectancyMonitor, EnvironmentState
)
from telemetry.slippage_monitor import SlippageMonitor

# Optional: real option chain fetcher (requires Alpaca clients)
try:
    from strategy.option_chain_fetcher import OptionChainFetcher, ContractInfo
    _CHAIN_FETCHER_OK = True
except ImportError:
    _CHAIN_FETCHER_OK = False

# High-vol config auto-selection
try:
    from strategy.v14_2_highvol_config import get_highvol_config, is_highvol_symbol
    _HIGHVOL_CONFIG_OK = True
except ImportError:
    _HIGHVOL_CONFIG_OK = False

# V14.3 symbol policy
try:
    from strategy.v14_3_highvol_config import (
        get_v14_3_config, get_symbol_policy, is_symbol_active,
        get_size_multiplier, can_trade_symbol, SYMBOL_POLICY_STATE,
    )
    _V14_3_OK = True
except ImportError:
    _V14_3_OK = False


class V14_2_CoreRunner:
    """
    Main strategy orchestrator for V14.2 Core Runner.
    
    Integrates with the existing AlpacaTrader's:
    - Trading client (position tracking, cash accounting)
    - Data feeds (underlying bar buffers)
    - Logging system
    
    Architecture:
    1. RouteConditioner evaluates price action → generates candidates
    2. WatchTicketBook creates watch tickets (WATCH-ONLY, no orders)
    3. ConfirmationEngine validates directional confirmation
    4. SpreadQualityGate validates option quote quality
    5. PackageBuilder constructs core-runner or V12.2 fallback
    6. RiskManager enforces hard limits
    7. MlegExecutionManager is the ONLY order writer
    8. Telemetry monitors adapt parameters
    """

    VERSION = "14.2"
    STATUS = "SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN"
    LABEL = "V14_2_CORE_RUNNER_REPLACEMENT_SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN"

    def __init__(
        self,
        trading_client: Any = None,
        option_data_client: Any = None,
        config: Optional[dict] = None,
        paper_mode: bool = True,
        log_dir: str = "HFT/logs/v14_2",
    ):
        self.cfg = config or get_config()
        self.paper_mode = paper_mode
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trading_client = trading_client
        self.option_data_client = option_data_client

        # ─── Safety: Live trading guard ──────────────────────────────────
        if not paper_mode and not self.cfg.get("live_trading_enabled", False):
            raise RuntimeError(
                "V14.2 Core Runner: live_trading_enabled is False. "
                "Cannot run in live mode. Set paper_mode=True or explicitly enable "
                "live_trading_enabled in config (NOT RECOMMENDED without real quote validation)."
            )

        # ─── Initialize Components ───────────────────────────────────────
        self.ticket_book = WatchTicketBook(log_dir=str(self.log_dir))
        self.confirmation_engine = ConfirmationEngine(config=self.cfg)
        self.spread_gate = SpreadQualityGate(config=self.cfg)
        self.package_builder = PackageBuilder(config=self.cfg)
        self.route_conditioner = RouteConditioner(config=self.cfg)
        self.risk_manager = RiskManager(config=self.cfg, log_dir=str(self.log_dir))

        self.execution_manager = MlegExecutionManager(
            trading_client=trading_client,
            config=self.cfg,
            log_dir=str(self.log_dir),
            paper_mode=paper_mode,
        )

        self.fill_logger = FillQualityLogger(log_dir=str(self.log_dir))
        self.route_monitor = RouteExpectancyMonitor(log_dir=str(self.log_dir))
        self.env_monitor = EnvironmentExpectancyMonitor(log_dir=str(self.log_dir))
        self.slippage_monitor = SlippageMonitor(log_dir=str(self.log_dir))

        # ─── Option Chain Fetcher (real quotes from Alpaca) ──────────────
        self.chain_fetcher = None
        if _CHAIN_FETCHER_OK and trading_client and option_data_client:
            self.chain_fetcher = OptionChainFetcher(trading_client, option_data_client)
            print("[V14.2] Real option chain fetcher: ENABLED")
        else:
            print("[V14.2] Real option chain fetcher: DISABLED (using simulated quotes)")

        # ─── State ───────────────────────────────────────────────────────
        self._current_environment = EnvironmentState(timestamp=datetime.utcnow())
        self._account_equity: float = 0.0
        self._buying_power: float = 0.0

        print(f"[V14.2] Core Runner initialized | mode={'PAPER' if paper_mode else 'LIVE'}")
        print(f"[V14.2] Status: {self.STATUS}")
        print(f"[V14.2] Live trading: {'ENABLED' if self.cfg.get('live_trading_enabled') else 'DISABLED'}")

    def update_account(self, equity: float, buying_power: float):
        """Update account state from the parent trader."""
        self._account_equity = equity
        self._buying_power = buying_power
        self.risk_manager.update_account(equity)

    def on_new_day(self):
        """Called at the start of each trading day."""
        self.risk_manager.reset_day()
        self.execution_manager.cancel_stale_orders()
        print(f"[V14.2] New day reset | {date.today()}")

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
    ) -> Dict[str, Any]:
        """
        Main entry point: evaluate a symbol for potential trade opportunity.
        
        This method implements the full V14.2 pipeline:
        1. Route scoring (WATCH-ONLY at base signal)
        2. Watch ticket creation
        3. Confirmation check on existing tickets
        4. Spread quality validation
        5. Package/fallback decision
        6. Risk check
        7. Order submission (via execution manager only)
        
        Returns a status dict describing what happened.
        """
        result = {
            "action": "NONE",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {},
        }

        # ─── V14.3 Symbol Policy Check ──────────────────────────────────
        if _V14_3_OK and self.cfg.get("version") == "14.3":
            allowed, reason, size_mult = can_trade_symbol(symbol)
            if not allowed:
                result["action"] = "SKIPPED"
                result["details"]["reason"] = f"symbol_policy: {reason}"
                return result
            result["details"]["size_multiplier"] = size_mult

        # ─── Phase 1: Route Evaluation (WATCH-ONLY) ──────────────────────
        candidates = self.route_conditioner.evaluate_routes(
            price=price, vwap=vwap, atr=atr,
            high_of_day=high_of_day, low_of_day=low_of_day,
            trend_slope=trend_slope, volume_ratio=volume_ratio,
            price_5m_ago=price_5m_ago, price_15m_ago=price_15m_ago,
            option_liquidity=option_liquidity, iv_percentile=iv_percentile,
        )

        # ─── Phase 2: Create watch tickets for qualifying candidates ─────
        for candidate in candidates:
            # Check if route is allowed by telemetry
            if not self.route_monitor.is_route_allowed(candidate.route):
                result["action"] = "SKIPPED"
                result["details"]["reason"] = f"route_disabled: {candidate.route}"
                continue

            # Check environment
            if not self.env_monitor.is_environment_allowed(self._current_environment):
                result["action"] = "SKIPPED"
                result["details"]["reason"] = "environment_blocked"
                continue

            # Create watch ticket (NEVER submits orders)
            side = TicketSide.CALL if candidate.side == "CALL" else TicketSide.PUT
            ticket = self.ticket_book.create_ticket(
                symbol=symbol,
                route=candidate.route,
                side=side,
                underlying_price_at_watch=price,
                vwap_at_watch=vwap,
                atr_at_watch=atr,
                route_score=candidate.score,
                ic_spread=candidate.ic_spread,
                expected_ev_over_debit=candidate.ev_over_debit,
                option_liquidity_score=candidate.option_liquidity,
                expected_move_to_target=candidate.expected_move,
                estimated_debit_at_watch=1.50,  # Placeholder; real value from option chain
            )

            if ticket:
                result["action"] = "WATCHING"
                result["details"]["ticket_id"] = ticket.ticket_id
                result["details"]["route"] = candidate.route
                result["details"]["score"] = candidate.score

        # ─── Phase 3: Check confirmations on existing watching tickets ───
        for ticket in self.ticket_book.get_watching_tickets(symbol):
            confirmation = self._check_ticket_confirmation(
                ticket, price, vwap, atr, high_of_day, low_of_day
            )

            if confirmation and confirmation.confirmed:
                self.confirmation_engine.apply_confirmation(
                    ticket, confirmation, timestamp=datetime.utcnow()
                )
                result["action"] = "CONFIRMED"
                result["details"]["ticket_id"] = ticket.ticket_id

                # ─── Phase 4: Spread Quality + Execution ─────────────────
                exec_result = self._attempt_execution(ticket, price, atr)
                if exec_result:
                    result["action"] = exec_result.get("action", "CONFIRMED")
                    result["details"].update(exec_result)

        # ─── Phase 5: Manage exits for filled positions ──────────────────
        self._manage_exits(symbol, price, atr)

        # ─── Phase 6: Expire stale tickets ───────────────────────────────
        self.ticket_book.expire_stale_tickets()

        return result

    def _check_ticket_confirmation(
        self,
        ticket: WatchTicket,
        price: float,
        vwap: float,
        atr: float,
        high_of_day: float,
        low_of_day: float,
    ) -> Optional[ConfirmationResult]:
        """Check if a watching ticket should be confirmed."""
        # Compute directional metrics
        if ticket.side == TicketSide.CALL:
            favorable_move = price - ticket.underlying_price_at_watch
        else:
            favorable_move = ticket.underlying_price_at_watch - price

        mfe_velocity = favorable_move / atr if atr > 0 else 0.0

        # Use high/low since watch
        high_since = max(high_of_day, ticket.underlying_price_at_watch)
        low_since = min(low_of_day, ticket.underlying_price_at_watch)

        return self.confirmation_engine.check_confirmation(
            ticket=ticket,
            current_price=price,
            current_vwap=vwap,
            current_atr=atr,
            high_since_watch=high_since,
            low_since_watch=low_since,
            mfe_velocity=mfe_velocity,
            env_stress=0.1,  # TODO: compute from environment monitor
            route_score_now=ticket.route_score,
            option_quote_valid=True,  # In paper mode, assume valid
        )

    def _attempt_execution(
        self, ticket: WatchTicket, underlying_price: float, atr: float
    ) -> Optional[Dict[str, Any]]:
        """
        After confirmation, validate spread quality and attempt execution.
        Uses NEXT available quote (not confirmation moment).
        Fetches REAL option quotes from Alpaca when chain_fetcher is available.
        """
        # ─── Risk Pre-Check ──────────────────────────────────────────────
        risk_check = self.risk_manager.pre_trade_check(
            symbol=ticket.symbol,
            route=ticket.route,
            debit=ticket.estimated_debit_at_watch * 100,
            is_live=not self.paper_mode,
        )

        if not risk_check.allowed:
            ticket.mark_rejected(f"risk: {risk_check.reason}")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            return {"action": "REJECTED", "reason": risk_check.reason}

        # ─── Fetch Real Option Quotes ────────────────────────────────────
        side = "CALL" if ticket.side == TicketSide.CALL else "PUT"
        long_leg = None
        short_leg = None
        long_contract = None
        short_contract = None

        if self.chain_fetcher:
            # REAL quotes from Alpaca
            result = self.chain_fetcher.get_spread_with_quotes(
                symbol=ticket.symbol,
                underlying_price=underlying_price,
                side=side,
                dte_min=5,
                dte_max=14,
                strike_width=5.0,
            )
            if result:
                long_leg, short_leg, long_contract, short_contract = result
                print(f"  [CHAIN] Real quotes: long={long_contract.symbol} "
                      f"bid={long_leg.bid:.2f} ask={long_leg.ask:.2f} | "
                      f"short={short_contract.symbol} "
                      f"bid={short_leg.bid:.2f} ask={short_leg.ask:.2f}")
            else:
                ticket.mark_rejected("no_option_contracts_found")
                self.ticket_book.retire_ticket(ticket.ticket_id)
                return {"action": "REJECTED", "reason": "no_option_contracts_available"}
        else:
            # Simulated quotes for testing without API connection
            long_leg = OptionLeg(
                contract_symbol=f"{ticket.symbol}_C_LONG",
                side="buy", bid=1.40, ask=1.60, mid=1.50,
                delta=0.50, iv=0.30, volume=100,
                open_interest=500, dte=7, strike=underlying_price,
            )
            short_leg = OptionLeg(
                contract_symbol=f"{ticket.symbol}_C_SHORT",
                side="sell", bid=0.80, ask=0.95, mid=0.875,
                delta=0.30, iv=0.32, volume=80,
                open_interest=300, dte=7, strike=underlying_price + 5.0,
            )

        # ─── Spread Quality Gate ─────────────────────────────────────────
        target_price = ticket.underlying_price_at_watch + ticket.expected_move_to_target
        if ticket.side == TicketSide.PUT:
            target_price = ticket.underlying_price_at_watch - ticket.expected_move_to_target

        quality = self.spread_gate.evaluate(
            long_leg=long_leg, short_leg=short_leg,
            underlying_price=underlying_price,
            underlying_price_at_watch=ticket.underlying_price_at_watch,
            estimated_debit_at_watch=ticket.estimated_debit_at_watch,
            target_price=target_price,
            route=ticket.route,
        )

        if not quality.passed:
            ticket.mark_rejected(f"spread_quality: {quality.rejection_reason}")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            self.slippage_monitor.record_entry_fill(0.0, was_filled=False)
            return {"action": "REJECTED", "reason": quality.rejection_reason}

        # ─── Package Decision ────────────────────────────────────────────
        package = self.package_builder.build(
            quality_report=quality,
            quality_score=ticket.route_score,
            pwin=0.60,  # TODO: from model
            env_stress=0.1,
            route=ticket.route,
            account_buying_power=self._buying_power,
        )

        # ─── Cancel stale entries before new submission ──────────────────
        self.execution_manager.cancel_all_entries_for_symbol(ticket.symbol)

        # ─── Submit via Execution Manager (REAL contracts if available) ──
        if long_contract and short_contract:
            # Use real contract symbols
            legs = [
                MlegLeg(
                    contract_symbol=long_contract.symbol,
                    side="buy_to_open",
                    quantity=package.total_contracts,
                    option_type=long_contract.option_type,
                    strike=long_contract.strike,
                    expiration=long_contract.expiration,
                ),
                MlegLeg(
                    contract_symbol=short_contract.symbol,
                    side="sell_to_open",
                    quantity=package.total_contracts,
                    option_type=short_contract.option_type,
                    strike=short_contract.strike,
                    expiration=short_contract.expiration,
                ),
            ]
        else:
            # Simulated contract symbols
            legs = [
                MlegLeg(
                    contract_symbol=long_leg.contract_symbol,
                    side="buy_to_open",
                    quantity=package.total_contracts,
                    option_type="call" if ticket.side == TicketSide.CALL else "put",
                    strike=long_leg.strike,
                ),
                MlegLeg(
                    contract_symbol=short_leg.contract_symbol,
                    side="sell_to_open",
                    quantity=package.total_contracts,
                    option_type="call" if ticket.side == TicketSide.CALL else "put",
                    strike=short_leg.strike,
                ),
            ]

        order = self.execution_manager.create_entry_order(
            ticket_id=ticket.ticket_id,
            legs=legs,
            limit_price=quality.spread_mid * 1.01,  # 1% above mid
            quantity=package.total_contracts,
            composite_bid=quality.spread_bid,
            composite_ask=quality.spread_ask,
            composite_mid=quality.spread_mid,
        )

        if order is None:
            ticket.mark_rejected("duplicate_or_contradictory_order")
            self.ticket_book.retire_ticket(ticket.ticket_id)
            return {"action": "REJECTED", "reason": "duplicate_order"}

        # Submit
        submitted = self.execution_manager.submit_order(order)
        if submitted and order.state == OrderState.FILLED:
            ticket.mark_filled()
            self.risk_manager.add_open_position(ticket.symbol, package.total_debit)
            self.risk_manager.record_trade(
                ticket.symbol, ticket.route, package.total_debit
            )
            self.slippage_monitor.record_entry_fill(
                order.fill_slippage_vs_mid, was_filled=True
            )
            return {
                "action": "FILLED",
                "mode": "PACKAGE" if package.is_package else "FALLBACK",
                "contracts": package.total_contracts,
                "debit": package.total_debit,
                "order_id": order.order_id,
                "broker_order_id": order.broker_order_id,
            }
        elif submitted and order.state == OrderState.SUBMITTED:
            # Order is pending with broker - will check status later
            return {
                "action": "SUBMITTED",
                "order_id": order.order_id,
                "broker_order_id": order.broker_order_id,
                "mode": "PACKAGE" if package.is_package else "FALLBACK",
            }
        elif order.state == OrderState.MISSED_FILL:
            ticket.mark_missed_fill("order_not_filled")
            self.slippage_monitor.record_entry_fill(0.0, was_filled=False)
            self.fill_logger.record_missed_fill(order.order_id, ticket.ticket_id, "timeout")
            return {"action": "MISSED_FILL", "reason": "order_timeout"}
        else:
            return {"action": "REJECTED", "reason": order.cancel_reason or "submission_failed"}

    def _manage_exits(self, symbol: str, price: float, atr: float):
        """
        Bot-managed exits for filled positions.
        All exits use closing mleg limit orders.
        """
        for ticket in self.ticket_book.get_confirmed_tickets(symbol):
            if ticket.status != TicketStatus.FILLED:
                continue

            orders = self.execution_manager.get_orders_for_ticket(ticket.ticket_id)
            entry_order = next(
                (o for o in orders if o.direction == "OPEN" and o.state == OrderState.FILLED),
                None,
            )
            if not entry_order:
                continue

            # Check if we already have a pending close
            pending_close = any(
                o.state == OrderState.CLOSE_SUBMITTED for o in orders if o.direction == "CLOSE"
            )
            if pending_close:
                continue

            # Evaluate exit conditions
            entry_price = entry_order.actual_fill_price
            if entry_price <= 0:
                continue

            # Simplified exit logic for paper mode
            # In production, monitor spread P&L vs targets
            if ticket.side == TicketSide.CALL:
                move_pct = (price - ticket.confirmation_price) / ticket.confirmation_price
            else:
                move_pct = (ticket.confirmation_price - price) / ticket.confirmation_price

            # Check stop
            if move_pct <= self.cfg["package_stop_pct"]:
                self._submit_exit(ticket, entry_order, "stop_hit")
            # Check target (core)
            elif move_pct >= self.cfg["core_target_pct"]:
                self._submit_exit(ticket, entry_order, "target_hit")

    def _submit_exit(self, ticket: WatchTicket, entry_order: MlegOrder, reason: str):
        """Submit a bot-managed closing mleg limit order."""
        # Reverse the legs for closing
        close_legs = []
        for leg in entry_order.legs:
            close_side = "sell_to_close" if "buy" in leg.side else "buy_to_close"
            close_legs.append(MlegLeg(
                contract_symbol=leg.contract_symbol,
                side=close_side,
                quantity=leg.quantity,
                option_type=leg.option_type,
                strike=leg.strike,
            ))

        # Conservative exit limit (at bid)
        exit_bid = entry_order.composite_bid  # Simplified
        exit_order = self.execution_manager.create_exit_order(
            ticket_id=ticket.ticket_id,
            legs=close_legs,
            limit_price=exit_bid,
            quantity=entry_order.quantity,
            exit_bid=exit_bid,
            exit_ask=entry_order.composite_ask,
        )

        if exit_order:
            self.execution_manager.submit_exit(exit_order)
            if exit_order.state == OrderState.CLOSED:
                # Calculate P&L
                pnl = (exit_order.actual_exit_price - entry_order.actual_fill_price) * 100
                ticket.mark_closed(pnl, reason)
                self.ticket_book.retire_ticket(ticket.ticket_id)
                self.risk_manager.record_close(ticket.symbol, pnl)
                self.route_monitor.record_trade(ticket.route, pnl)
                self.env_monitor.record_trade(
                    self._current_environment, ticket.symbol, pnl
                )
                self.slippage_monitor.record_exit_fill(exit_order.exit_slippage_vs_bid)

    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status for logging/display."""
        return {
            "version": self.VERSION,
            "status": self.STATUS,
            "label": self.LABEL,
            "mode": "PAPER" if self.paper_mode else "LIVE",
            "live_enabled": self.cfg.get("live_trading_enabled", False),
            "active_watches": len(self.ticket_book.active_tickets),
            "risk_killed": self.risk_manager.is_killed(),
            "slippage_degraded": self.slippage_monitor.config.is_degraded,
            "route_stats": self.route_monitor.get_all_route_stats(),
        }


def create_runner_for_symbol(
    symbol: str,
    trading_client: Any = None,
    option_data_client: Any = None,
    paper_mode: bool = True,
    log_dir: str = "HFT/logs/v14_2",
) -> V14_2_CoreRunner:
    """
    Factory: create a V14.2/V14.3 runner with auto-selected config.

    V14.3 high-vol profile: COIN, TSLA (symbol-policy gated)
    V14.2 high-vol legacy: other high-vol names in the list
    V14.2 base: everything else (SPY, QQQ, AAPL, etc.)
    """
    # V14.3 takes priority for its allowed symbols
    if _V14_3_OK and is_symbol_active(symbol):
        config = get_v14_3_config()
        print(f"[V14.3] Using HIGH-VOL LOCAL OPTIMUM config for {symbol}")
        policy = get_symbol_policy(symbol)
        print(f"[V14.3] Policy: {policy['mode']} | size={policy.get('size_multiplier', 1.0)}")
    elif _HIGHVOL_CONFIG_OK and is_highvol_symbol(symbol):
        config = get_highvol_config()
        print(f"[V14.2-HV] Using high-vol config for {symbol}")
    else:
        config = get_config()
        print(f"[V14.2] Using BASE config for {symbol}")

    return V14_2_CoreRunner(
        trading_client=trading_client,
        option_data_client=option_data_client,
        config=config,
        paper_mode=paper_mode,
        log_dir=log_dir,
    )
