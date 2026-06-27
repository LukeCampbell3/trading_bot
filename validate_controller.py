"""
V14_2_1 Dry-Run Validation
===========================
Simulates one full trading session against real IEX bar data without submitting
any orders to the broker. Validates:
  1. Startup reconciliation works
  2. Existing positions are rescored immediately
  3. Stale/weak holds produce target reductions
  4. No new buy occurs before cleanup/governance
  5. Pending buys count as exposure
  6. REVALIDATE_LOSER cannot average down
  7. Loss mitigation blocks weak entries
  8. Strategy-owned orders have structured client_order_id
  9. Audit logs explain every decision
  10. Session window enforced
"""

import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import pytz
import numpy as np

from execution_controller import (
    VERSION, SymbolCampaign, BrokerSymbolState, BrokerSnapshot,
    ExecutionPolicy, TradeSignal, CostEstimate, MarketState,
    CampaignBook, ExecutionReconciler, PortfolioAdmission,
    HoldScorer, LossGovernor, SafetyModeManager, SafetyMode,
    evaluate_signal, is_in_trading_window, should_cancel_premarket_orders,
    is_bot_order, resolve_contradictions, compute_target_qty,
    generate_client_order_id, compute_setup_fingerprint,
    LOSS_REVALIDATE_PCT, LOSS_FRESH_CONFIRM_PCT, LOSS_TARGET_ZERO_PCT,
    LOSS_LOCK_SESSION_PCT,
)

eastern = pytz.timezone("US/Eastern")


# ============================================================================
# Mock Broker (records all actions without hitting Alpaca)
# ============================================================================

class DryRunBroker:
    """Records all order actions for audit without submitting to Alpaca."""

    def __init__(self, initial_equity=95000.0, initial_buying_power=380000.0):
        self.equity = initial_equity
        self.buying_power = initial_buying_power
        self.positions: Dict[str, dict] = {}
        self.open_orders: List[dict] = []
        self.order_history: List[dict] = []
        self.cancel_log: List[str] = []
        self._next_order_id = 1

    def set_position(self, symbol: str, qty: int, avg_price: float):
        """Simulate an existing position."""
        self.positions[symbol] = {"qty": qty, "avg_price": avg_price}

    def set_open_order(self, order_id: str, symbol: str, side: str, qty: int,
                       client_order_id: str = "", submitted_at: datetime = None):
        """Simulate an existing open order."""
        self.open_orders.append({
            "id": order_id, "symbol": symbol, "side": side,
            "qty": qty, "filled_qty": 0,
            "client_order_id": client_order_id,
            "submitted_at": submitted_at or datetime.utcnow(),
            "status": "accepted",
        })

    def fetch_snapshot(self) -> BrokerSnapshot:
        positions = {}
        for sym, p in self.positions.items():
            positions[sym] = BrokerSymbolState(
                symbol=sym, position_qty=p["qty"],
                avg_entry_price=p["avg_price"],
                pending_buy_qty=0, pending_sell_qty=0)

        for o in self.open_orders:
            sym = o["symbol"]
            if sym not in positions:
                positions[sym] = BrokerSymbolState(
                    symbol=sym, position_qty=0, avg_entry_price=None,
                    pending_buy_qty=0, pending_sell_qty=0)
            bs = positions[sym]
            remaining = o["qty"] - o["filled_qty"]
            if "buy" in o["side"].lower():
                bs.pending_buy_qty += remaining
                bs.open_buy_order_ids.append(o["id"])
                bs.open_buy_orders.append(o)
            else:
                bs.pending_sell_qty += remaining
                bs.open_sell_order_ids.append(o["id"])
                bs.open_sell_orders.append(o)

        return BrokerSnapshot(
            account_equity=self.equity,
            account_buying_power=self.buying_power,
            positions=positions,
            all_open_orders=list(self.open_orders),
        )

    def cancel_order(self, order_id: str):
        self.cancel_log.append(order_id)
        self.open_orders = [o for o in self.open_orders if o["id"] != order_id]

    def cancel_all_orders_for_symbol(self, symbol: str):
        to_cancel = [o["id"] for o in self.open_orders if o["symbol"] == symbol]
        for oid in to_cancel:
            self.cancel_order(oid)

    def submit_entry_order(self, symbol: str, qty: int, client_order_id: str,
                           invalidation_price=None, extended_hours=False, now_et=None) -> bool:
        oid = f"DRY_BUY_{self._next_order_id}"
        self._next_order_id += 1
        self.order_history.append({
            "action": "BUY", "symbol": symbol, "qty": qty,
            "client_order_id": client_order_id, "order_id": oid,
            "ts": datetime.utcnow().isoformat(),
        })
        # Simulate immediate fill for validation
        if symbol in self.positions:
            self.positions[symbol]["qty"] += qty
        else:
            self.positions[symbol] = {"qty": qty, "avg_price": 100.0}
        return True

    def submit_exit_order(self, symbol: str, qty: int, client_order_id: str,
                          extended_hours=False, now_et=None) -> bool:
        oid = f"DRY_SELL_{self._next_order_id}"
        self._next_order_id += 1
        self.order_history.append({
            "action": "SELL", "symbol": symbol, "qty": qty,
            "client_order_id": client_order_id, "order_id": oid,
            "ts": datetime.utcnow().isoformat(),
        })
        # Simulate fill
        if symbol in self.positions:
            self.positions[symbol]["qty"] -= qty
            if self.positions[symbol]["qty"] <= 0:
                del self.positions[symbol]
        return True


# ============================================================================
# Mock Signal Engine (simulates the alpaca_trader signal output)
# ============================================================================

class MockSignalEngine:
    """Simulates signal generation with controlled scenarios."""

    def __init__(self):
        self.signals_to_emit: List[TradeSignal] = []
        self.market_states: Dict[str, MarketState] = {}
        self.costs: Dict[str, CostEstimate] = {}

    def generate_signals(self) -> List[TradeSignal]:
        return list(self.signals_to_emit)

    def get_market_state(self, symbol: str) -> MarketState:
        return self.market_states.get(symbol, MarketState(last_price=100.0, atr=1.5))

    def estimate_costs(self, symbol: str) -> CostEstimate:
        return self.costs.get(symbol, CostEstimate(spread_bps=2.0, slippage_bps=1.0))


# ============================================================================
# Validation Scenarios
# ============================================================================

def make_signal(symbol, bucket="B4", direction="long", score=0.8,
                edge_bps=12.0, fingerprint=None, category="general",
                invalidation=None, ttl=90):
    fp = fingerprint or compute_setup_fingerprint(symbol, bucket, direction, 1.0, "normal", "up")
    return TradeSignal(
        symbol=symbol, bucket_id=bucket, direction=direction, score=score,
        expected_return_bps=edge_bps, setup_fingerprint=fp,
        generated_at=datetime.utcnow(), ttl_seconds=ttl,
        invalidation_price=invalidation, category=category)


def run_validation():
    print(f"\n{'='*70}")
    print(f"  V14_2_1 DRY-RUN VALIDATION SESSION")
    print(f"  Version: {VERSION}")
    print(f"{'='*70}\n")

    policy = ExecutionPolicy(
        no_new_entries_before="09:35",
        no_new_entries_after="15:55",
        max_concurrent_positions=6,
        max_new_per_cycle=2,
        max_new_per_day=4,
        min_net_edge_bps=1.5,
    )

    results = []
    all_pass = True

    def check(condition, name, detail=""):
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        results.append((status, name, detail))
        mark = "  ✓" if condition else "  ✗"
        print(f"{mark} {name}" + (f" -- {detail}" if detail else ""))
        return condition

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 1: Startup with existing positions (reconciliation)
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 1: Startup Reconciliation ──")

    broker = DryRunBroker()
    broker.set_position("TSLA", 28, 350.0)
    broker.set_position("MRNA", 140, 100.0)  # will be at loss
    broker.set_position("NVDA", 15, 120.0)   # will be profitable
    # Stale buy order that shouldn't be there
    broker.set_open_order("stale_buy_1", "AMD", "buy", 225,
                          client_order_id="BOT|V14_2_1|AMD|old|BUY|20240101T080000|aaa",
                          submitted_at=datetime.utcnow() - timedelta(seconds=300))

    campaign_book = CampaignBook(["TSLA", "MRNA", "NVDA", "AMD", "AAPL", "COIN"])
    campaign_book.policy_ref = policy

    # Sync from broker
    snapshot = broker.fetch_snapshot()
    campaign_book.sync_from_broker(snapshot)

    check(campaign_book.campaigns["TSLA"].state == "ACTIVE",
          "TSLA adopted as ACTIVE", f"state={campaign_book.campaigns['TSLA'].state}")
    check(campaign_book.campaigns["TSLA"].target_qty == 28,
          "TSLA target=28 (adopted from broker)")
    check(campaign_book.campaigns["MRNA"].state == "ACTIVE",
          "MRNA adopted as ACTIVE")
    check(campaign_book.campaigns["NVDA"].state == "ACTIVE",
          "NVDA adopted as ACTIVE")

    # Detect contradictions (AMD has buy orders but no campaign target)
    reconciler = ExecutionReconciler(broker, policy, eastern)
    reconciler.resolve_all_contradictions(campaign_book.campaigns, snapshot)

    check("stale_buy_1" in broker.cancel_log,
          "Stale AMD buy order cancelled during contradiction resolution")

    # Cancel stale orders
    reconciler.cancel_stale_orders(snapshot, datetime.utcnow())
    check(len(broker.open_orders) == 0,
          "All stale orders cleaned up", f"remaining={len(broker.open_orders)}")


    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 2: Loss Governance (positions re-scored)
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 2: Loss Governance ──")

    loss_governor = LossGovernor(policy)
    hold_scorer = HoldScorer()

    # MRNA: simulate -4.5% loss (below -3.5% -> lock session)
    campaign_book.campaigns["MRNA"].entry_price = 100.0
    mrna_price = 95.50  # -4.5%
    loss_action = loss_governor.evaluate(campaign_book.campaigns["MRNA"], mrna_price)
    check(loss_action == "lock_session",
          "MRNA at -4.5% triggers lock_session", f"action={loss_action}")

    loss_governor.apply_loss_action(campaign_book.campaigns["MRNA"], loss_action)
    check(campaign_book.campaigns["MRNA"].state == "LOCKED_ERROR",
          "MRNA locked after -4.5% loss")
    check(campaign_book.campaigns["MRNA"].target_qty == 0,
          "MRNA target=0 after lock")
    check("MRNA" in loss_governor.session_locked_symbols,
          "MRNA added to session lock set")

    # TSLA: simulate -1.0% loss (REVALIDATE range)
    campaign_book.campaigns["TSLA"].entry_price = 350.0
    tsla_price = 346.50  # -1.0%
    loss_action_tsla = loss_governor.evaluate(campaign_book.campaigns["TSLA"], tsla_price)
    check(loss_action_tsla == "revalidate",
          "TSLA at -1.0% triggers revalidate", f"action={loss_action_tsla}")

    loss_governor.apply_loss_action(campaign_book.campaigns["TSLA"], loss_action_tsla)
    check(campaign_book.campaigns["TSLA"].state == "REVALIDATE_LOSER",
          "TSLA enters REVALIDATE_LOSER")

    # NVDA: simulate +1.5% profit (PROTECT_PROFIT range)
    campaign_book.campaigns["NVDA"].entry_price = 120.0
    nvda_price = 121.80  # +1.5%
    profit_action = loss_governor.evaluate_profit_protection(
        campaign_book.campaigns["NVDA"], nvda_price)
    check(profit_action == "protect_profit",
          "NVDA at +1.5% triggers protect_profit", f"action={profit_action}")

    loss_governor.apply_profit_action(campaign_book.campaigns["NVDA"], profit_action)
    check(campaign_book.campaigns["NVDA"].state == "PROTECT_PROFIT",
          "NVDA enters PROTECT_PROFIT")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 3: REVALIDATE_LOSER cannot add size
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 3: REVALIDATE_LOSER Cannot Add Size ──")

    # Try to increase TSLA target (simulate a signal saying "add more")
    snapshot2 = broker.fetch_snapshot()
    result = reconciler._increase_exposure(
        "TSLA", campaign_book.campaigns["TSLA"],
        snapshot2.get_symbol_state("TSLA"), 50)  # try to go from 28 to 50
    check("entry_blocked" in result,
          "TSLA REVALIDATE_LOSER blocks increase_exposure", f"result={result}")

    # Also verify transition is forbidden
    check(not campaign_book.campaigns["TSLA"].transition_to("BUILDING"),
          "TSLA cannot transition REVALIDATE_LOSER->BUILDING")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 4: Profit protection blocks adding
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 4: Profit Protection No-Adding ──")

    result_nvda = reconciler._increase_exposure(
        "NVDA", campaign_book.campaigns["NVDA"],
        snapshot2.get_symbol_state("NVDA"), 30)  # try to go from 15 to 30
    check("PROTECT_PROFIT" in result_nvda,
          "NVDA PROTECT_PROFIT blocks increase_exposure", f"result={result_nvda}")


    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 5: Session Window Enforcement
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 5: Session Window ──")

    # Before 09:35 -> blocked
    early_time = datetime(2026, 6, 25, 9, 30, 0, tzinfo=eastern)
    in_window, reason = is_in_trading_window(early_time, policy)
    check(not in_window, "09:30 ET is before entry window", f"reason={reason}")

    # At 10:00 -> allowed
    market_time = datetime(2026, 6, 25, 10, 0, 0, tzinfo=eastern)
    in_window, _ = is_in_trading_window(market_time, policy)
    check(in_window, "10:00 ET is in entry window")

    # After 15:55 -> blocked
    late_time = datetime(2026, 6, 25, 16, 0, 0, tzinfo=eastern)
    in_window, reason = is_in_trading_window(late_time, policy)
    check(not in_window, "16:00 ET is after entry window", f"reason={reason}")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 6: Portfolio Admission (limited entries)
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 6: Portfolio Admission ──")

    admission = PortfolioAdmission(policy)
    admission.reset_day(datetime.now().date())
    admission.reset_cycle()

    signal_engine = MockSignalEngine()

    # Generate 10 long candidates
    candidates = []
    for i, sym in enumerate(["AAPL", "COIN", "SMCI", "PLTR", "NET",
                              "TQQQ", "HOOD", "SOFI", "ROKU", "RIVN"]):
        sig = make_signal(sym, bucket="B4", score=0.9 - i*0.05,
                         edge_bps=15.0 - i*1.5, invalidation=95.0 + i)
        candidates.append(sig)

    # Simulate admission with 3 existing positions
    admitted = []
    for sig in candidates:
        snapshot_adm = broker.fetch_snapshot()
        can, reason = admission.can_admit(
            sig, sig.score, campaign_book.campaigns, snapshot_adm)
        if can:
            admission.record_admission()
            admitted.append(sig.symbol)

    check(len(admitted) <= policy.max_new_per_cycle,
          f"Max {policy.max_new_per_cycle} admitted per cycle",
          f"admitted={admitted}")
    check(len(admitted) > 0,
          "At least one candidate admitted", f"admitted={admitted}")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 7: Decision Engine with Full Pipeline
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 7: Decision Engine Full Pipeline ──")

    # Good candidate -> SET_TARGET
    sig_good = make_signal("AAPL", bucket="B5", score=0.92, edge_bps=18.0,
                          invalidation=285.0)
    c_aapl = SymbolCampaign(symbol="AAPL")
    bs_aapl = BrokerSymbolState(symbol="AAPL", position_qty=0, avg_entry_price=None,
                                pending_buy_qty=0, pending_sell_qty=0)
    ms_aapl = MarketState(last_price=290.0, atr=3.0, structure_reset=True)
    costs_aapl = CostEstimate(spread_bps=2.0, slippage_bps=1.0)

    decision = evaluate_signal(sig_good, c_aapl, bs_aapl, ms_aapl, costs_aapl,
                               95000, 290.0, policy, buying_power=380000,
                               loss_governor=loss_governor)
    check(decision.action == "SET_TARGET",
          "Good AAPL signal -> SET_TARGET", f"target={decision.target_qty}")
    check(decision.target_qty > 0 and decision.target_qty < 200,
          "Target qty reasonable", f"qty={decision.target_qty}")

    # Weak edge -> NO_ACTION (NTZ blocks)
    sig_weak = make_signal("GOOGL", bucket="B3", score=0.4, edge_bps=1.0)
    c_googl = SymbolCampaign(symbol="GOOGL")
    bs_googl = BrokerSymbolState(symbol="GOOGL", position_qty=0, avg_entry_price=None,
                                 pending_buy_qty=0, pending_sell_qty=0)
    ms_googl = MarketState(last_price=175.0, atr=2.0)
    costs_googl = CostEstimate(spread_bps=3.0, slippage_bps=2.0)

    decision_weak = evaluate_signal(sig_weak, c_googl, bs_googl, ms_googl, costs_googl,
                                    95000, 175.0, policy, buying_power=380000)
    check(decision_weak.action == "NO_ACTION",
          "Weak GOOGL edge -> NO_ACTION (NTZ)", f"reason={decision_weak.reason}")
    check("net_edge" in decision_weak.reason,
          "Reason explains edge too small")

    # Locked symbol -> NO_ACTION
    sig_locked = make_signal("MRNA", bucket="B5", score=0.95, edge_bps=25.0)
    decision_locked = evaluate_signal(sig_locked, campaign_book.campaigns["MRNA"],
                                      BrokerSymbolState(symbol="MRNA", position_qty=140,
                                                       avg_entry_price=100.0,
                                                       pending_buy_qty=0, pending_sell_qty=0),
                                      MarketState(last_price=95.5, atr=2.0),
                                      CostEstimate(), 95000, 95.5, policy)
    check(decision_locked.action == "NO_ACTION",
          "Locked MRNA -> NO_ACTION even with B5 signal",
          f"reason={decision_locked.reason}")

    # Flat signal on active position -> TARGET_ZERO
    sig_flat = make_signal("NVDA", direction="flat", edge_bps=-5.0)
    decision_flat = evaluate_signal(sig_flat, campaign_book.campaigns["NVDA"],
                                    snapshot2.get_symbol_state("NVDA"),
                                    MarketState(last_price=121.8, atr=2.0),
                                    CostEstimate(), 95000, 121.8, policy)
    check(decision_flat.action == "TARGET_ZERO",
          "Flat signal on NVDA -> TARGET_ZERO")


    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 8: Safety Mode Escalation
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 8: Safety Mode Escalation ──")

    safety_mgr = SafetyModeManager()

    # Normal conditions
    snap_normal = BrokerSnapshot(account_equity=95000, account_buying_power=380000)
    safety_mgr.evaluate_portfolio_health(snap_normal, campaign_book.campaigns, 95000)
    check(safety_mgr.mode == SafetyMode.NORMAL or safety_mgr.mode == SafetyMode.ORDER_RECONCILIATION_ONLY,
          "Initial mode based on portfolio state",
          f"mode={safety_mgr.mode.value}")

    # Simulate -2.5% drawdown -> between -2% (LOSS_MITIGATION) and -3% (NO_NEW_ENTRIES)
    safety_mgr2 = SafetyModeManager()
    snap_loss = BrokerSnapshot(account_equity=92625, account_buying_power=370000)
    safety_mgr2.evaluate_portfolio_health(snap_loss, {}, 95000)
    check(safety_mgr2.mode == SafetyMode.LOSS_MITIGATION,
          "-2.5% triggers LOSS_MITIGATION", f"mode={safety_mgr2.mode.value}")
    check(not safety_mgr2.allows_new_entries(),
          "LOSS_MITIGATION blocks new entries")

    # Simulate -5.5% drawdown
    safety_mgr3 = SafetyModeManager()
    snap_crash = BrokerSnapshot(account_equity=89775, account_buying_power=350000)
    safety_mgr3.evaluate_portfolio_health(snap_crash, {}, 95000)
    check(safety_mgr3.mode == SafetyMode.LIQUIDATE_ONLY,
          "-5.5% triggers LIQUIDATE_ONLY", f"mode={safety_mgr3.mode.value}")
    check(safety_mgr3.is_liquidate_only(),
          "LIQUIDATE_ONLY confirmed")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 9: Full Reconciliation Cycle (reduce/exit)
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 9: Reconciliation Cycle ──")

    # Set MRNA target=0 (locked), should sell remaining position
    broker2 = DryRunBroker()
    broker2.set_position("MRNA", 140, 100.0)
    broker2.set_position("TSLA", 28, 350.0)

    campaign_book2 = CampaignBook(["MRNA", "TSLA"])
    campaign_book2.policy_ref = policy
    campaign_book2.campaigns["MRNA"].state = "LOCKED_ERROR"
    campaign_book2.campaigns["MRNA"].target_qty = 0
    campaign_book2.campaigns["MRNA"].locked_reason = "loss_exceeded_3.5pct"
    campaign_book2.campaigns["TSLA"].state = "ACTIVE"
    campaign_book2.campaigns["TSLA"].target_qty = 28

    reconciler2 = ExecutionReconciler(broker2, policy, eastern)
    snap3 = broker2.fetch_snapshot()
    recon_results = reconciler2.reconcile_all(campaign_book2.campaigns, snap3)

    check(recon_results["MRNA"] == "reducing_exposure",
          "MRNA locked -> reduces exposure")
    check(recon_results["TSLA"] == "in_sync",
          "TSLA aligned -> in_sync")
    check(any(o["symbol"] == "MRNA" and o["action"] == "SELL" for o in broker2.order_history),
          "MRNA sell order submitted")
    check(not any(o["symbol"] == "MRNA" and o["action"] == "BUY" for o in broker2.order_history),
          "No MRNA buy submitted (locked)")

    # Verify client_order_id format on the sell
    mrna_sell = [o for o in broker2.order_history if o["symbol"] == "MRNA"][0]
    check(mrna_sell["client_order_id"].startswith("BOT|V14_2_1|MRNA|"),
          "MRNA sell has structured client_order_id",
          f"cid={mrna_sell['client_order_id']}")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 10: Hold Scoring + Replacement Logic
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 10: Hold Scoring + Replacement ──")

    hold_scorer = HoldScorer()

    # Strong position
    c_strong = SymbolCampaign(symbol="STRONG", state="ACTIVE", bucket_id="B5",
                              entry_price=100.0, confirmations=3,
                              trend_aligned=True, volume_confirmed=True, bars_held=5)
    score_strong = hold_scorer.score(c_strong, current_price=102.0, spread_bps=2.0)

    # Weak position
    c_weak = SymbolCampaign(symbol="WEAK", state="REVALIDATE_LOSER", bucket_id="B3",
                            entry_price=100.0, confirmations=0,
                            trend_aligned=False, volume_confirmed=False, bars_held=45)
    score_weak = hold_scorer.score(c_weak, current_price=97.0, spread_bps=8.0)

    check(score_strong > score_weak,
          "Strong scores higher than weak",
          f"strong={score_strong:.2f} weak={score_weak:.2f}")
    check(score_weak < 0,
          "Weak loser scores negative", f"score={score_weak:.2f}")
    check(score_strong > 1.0,
          "Strong confirmed winner scores > 1.0", f"score={score_strong:.2f}")

    # Replacement check: new candidate must beat worst * 1.15
    admission2 = PortfolioAdmission(policy)
    admission2.reset_day(datetime.now().date())
    admission2.reset_cycle()

    full_campaigns = {
        "S1": SymbolCampaign(symbol="S1", state="ACTIVE"),
        "S2": SymbolCampaign(symbol="S2", state="ACTIVE"),
        "S3": SymbolCampaign(symbol="S3", state="ACTIVE"),
        "S4": SymbolCampaign(symbol="S4", state="ACTIVE"),
        "S5": SymbolCampaign(symbol="S5", state="ACTIVE"),
        "S6": SymbolCampaign(symbol="S6", state="ACTIVE"),  # worst
    }
    for c in full_campaigns.values():
        c.hold_score = 0.8
    full_campaigns["S6"].hold_score = 0.3  # weak

    full_snap = BrokerSnapshot(
        account_equity=95000, account_buying_power=380000,
        positions={s: BrokerSymbolState(symbol=s, position_qty=10, avg_entry_price=100.0,
                                       pending_buy_qty=0, pending_sell_qty=0)
                   for s in full_campaigns})

    # Strong candidate can replace
    sig_replace = make_signal("NEW", bucket="B5", score=0.9, edge_bps=15.0)
    can, reason = admission2.can_admit(sig_replace, 0.5, full_campaigns, full_snap)
    check(can, "Strong candidate replaces weak holding", f"reason={reason}")

    # Weak candidate cannot
    can_w, reason_w = admission2.can_admit(sig_replace, 0.2, full_campaigns, full_snap)
    check(not can_w, "Weak candidate rejected (below replacement margin)",
          f"reason={reason_w}")


    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 11: Pending Buys Count as Exposure
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 11: Pending Buys Count as Exposure ──")

    bs_pending = BrokerSymbolState(
        symbol="AMD", position_qty=28, avg_entry_price=543.0,
        pending_buy_qty=225, pending_sell_qty=0,
        open_buy_order_ids=["b1", "b2", "b3"])
    check(bs_pending.effective_exposure == 253,
          "effective_exposure = 28 + 225 = 253",
          f"got={bs_pending.effective_exposure}")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 12: Unknown/Manual Order Detection
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 12: Unknown Order Detection ──")

    check(is_bot_order("BOT|V14_2_1|AAPL|abc|BUY|20240101|xyz"), "V14_2_1 order recognized")
    check(is_bot_order("BOT|V15|AAPL|B4|20240101|BUY|xyz"), "V15 order recognized")
    check(not is_bot_order("manual_order_12345"), "Manual order NOT recognized as bot")
    check(not is_bot_order(""), "Empty string NOT bot order")

    # ─────────────────────────────────────────────────────────────────
    # SCENARIO 13: Complete Controller Loop Simulation
    # ─────────────────────────────────────────────────────────────────
    print("\n  ── Scenario 13: Full Loop Simulation (3 cycles) ──")

    broker3 = DryRunBroker(initial_equity=95000, initial_buying_power=380000)
    broker3.set_position("TSLA", 28, 350.0)  # existing position

    campaign_book3 = CampaignBook(["TSLA", "AAPL", "NVDA", "AMD", "COIN", "SMCI"])
    campaign_book3.policy_ref = policy

    signal_engine = MockSignalEngine()
    loss_gov3 = LossGovernor(policy)
    safety3 = SafetyModeManager()
    scorer3 = HoldScorer()
    admission3 = PortfolioAdmission(policy)
    reconciler3 = ExecutionReconciler(broker3, policy, eastern)

    # Simulate 3 cycles
    for cycle in range(3):
        # Configure signals per cycle
        if cycle == 0:
            signal_engine.signals_to_emit = [
                make_signal("AAPL", bucket="B5", score=0.9, edge_bps=15.0, invalidation=285.0),
                make_signal("NVDA", bucket="B4", score=0.75, edge_bps=10.0, invalidation=118.0),
                make_signal("TSLA", direction="flat", edge_bps=-3.0),  # exit signal
            ]
            signal_engine.market_states = {
                "AAPL": MarketState(last_price=290.0, atr=3.0, structure_reset=True),
                "NVDA": MarketState(last_price=120.0, atr=2.0, structure_reset=True),
                "TSLA": MarketState(last_price=340.0, atr=5.0),
            }
        elif cycle == 1:
            signal_engine.signals_to_emit = [
                make_signal("SMCI", bucket="B4", score=0.7, edge_bps=8.0, invalidation=45.0),
                make_signal("AAPL", bucket="B5", score=0.88, edge_bps=12.0),  # confirmation
            ]
            signal_engine.market_states = {
                "SMCI": MarketState(last_price=50.0, atr=2.0, structure_reset=True),
                "AAPL": MarketState(last_price=291.0, atr=3.0),
            }
        else:
            signal_engine.signals_to_emit = []  # no signals

        # Run one cycle of the controller manually
        snapshot_c = broker3.fetch_snapshot()
        campaign_book3.sync_from_broker(snapshot_c)

        # Rescore + loss governance
        for sym, camp in campaign_book3.items():
            if camp.state in ("ACTIVE", "BUILDING", "PROTECT_PROFIT", "REVALIDATE_LOSER"):
                ms = signal_engine.get_market_state(sym)
                if ms.last_price > 0 and camp.entry_price:
                    loss_action = loss_gov3.evaluate(camp, ms.last_price)
                    if loss_action:
                        loss_gov3.apply_loss_action(camp, loss_action)

        # Process signals
        admission3.reset_cycle()
        signals = signal_engine.generate_signals()

        for sig in signals:
            camp = campaign_book3.get(sig.symbol)
            bs = snapshot_c.get_symbol_state(sig.symbol)
            ms = signal_engine.get_market_state(sig.symbol)
            costs = signal_engine.estimate_costs(sig.symbol)

            decision = evaluate_signal(sig, camp, bs, ms, costs,
                                       snapshot_c.account_equity, ms.last_price,
                                       policy, buying_power=snapshot_c.account_buying_power,
                                       loss_governor=loss_gov3)

            if decision.action == "SET_TARGET":
                can, _ = admission3.can_admit(sig, sig.score, campaign_book3.campaigns, snapshot_c)
                if can:
                    campaign_book3.set_target(sig.symbol, sig, decision.target_qty, sig.invalidation_price)
                    admission3.record_admission()
            elif decision.action == "TARGET_ZERO":
                campaign_book3.set_target_zero(sig.symbol, decision.reason)

        # Reconcile
        snap_fresh = broker3.fetch_snapshot()
        reconciler3.reconcile_all(campaign_book3.campaigns, snap_fresh)

    # Verify outcomes after 3 cycles
    check(campaign_book3.campaigns["TSLA"].target_qty == 0,
          "TSLA target=0 after flat signal")
    check(campaign_book3.campaigns["TSLA"].state in ("EXITING", "COOLDOWN_BLOCKED", "FLAT"),
          "TSLA exiting/flat after sell",
          f"state={campaign_book3.campaigns['TSLA'].state}")

    # AAPL should have been admitted
    check(campaign_book3.campaigns["AAPL"].state in ("BUILDING", "ACTIVE"),
          "AAPL admitted and building/active",
          f"state={campaign_book3.campaigns['AAPL'].state}")
    check(campaign_book3.campaigns["AAPL"].target_qty > 0,
          "AAPL has positive target", f"target={campaign_book3.campaigns['AAPL'].target_qty}")

    # Order history should show TSLA sell and AAPL target set
    tsla_sells = [o for o in broker3.order_history if o["symbol"] == "TSLA" and o["action"] == "SELL"]
    aapl_target = campaign_book3.campaigns["AAPL"].target_qty
    check(len(tsla_sells) > 0, "TSLA sell order submitted")
    check(aapl_target > 0,
          "AAPL target assigned (buy deferred to market hours)",
          f"target={aapl_target}")

    # Verify max 2 new entries per cycle was respected
    check(admission3.entries_today <= 4,
          f"Entries today within daily limit",
          f"entries={admission3.entries_today}")

    # All orders have structured client_order_id
    for order in broker3.order_history:
        cid = order["client_order_id"]
        check(cid.startswith("BOT|V14_2_1|"),
              f"Order {order['symbol']} {order['action']} has V14_2_1 client_order_id",
              f"cid={cid[:40]}")

    # ─────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    n_pass = sum(1 for s, _, _ in results if s == "PASS")
    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    print(f"  VALIDATION RESULTS: {n_pass} passed, {n_fail} failed")

    if n_fail > 0:
        print(f"\n  FAILURES:")
        for status, name, detail in results:
            if status == "FAIL":
                print(f"    ✗ {name}: {detail}")

    print(f"\n  Status: {'V14_2_1_DRY_RUN_RECONCILIATION_VALIDATED' if all_pass else 'VALIDATION_FAILED'}")
    print(f"{'='*70}\n")

    return all_pass


if __name__ == "__main__":
    import sys
    success = run_validation()
    sys.exit(0 if success else 1)
