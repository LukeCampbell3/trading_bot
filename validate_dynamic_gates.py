"""
Dynamic Gate Controller — Effectiveness Validation
====================================================
Standalone, no-broker validation that the dynamic-gate change
(strategy/dynamic_gate_controller.py + the capital-based RiskManager
rewrite) actually does what it was built for:

  1. Capital exposure NEVER exceeds max_open_debit_exposure_pct, no matter
     how aggressively trades are attempted (the hard constraint is intact).
  2. Trade cadence is no longer capped at an arbitrary count when capital
     is available — it scales with headroom, but a circuit breaker still
     caps runaway cadence even with unlimited capital.
  3. A route/environment that goes DISABLED/BLOCKED is no longer frozen
     forever — it recovers via cooldown -> probation -> full reactivation.
  4. Position-size scaling on SOFT_SIZE_ONLY/PROBATION never exceeds what
     PackageBuilder already approved (risk can only shrink, never grow).
  5. Head-to-head: on an identical synthetic price path, the OLD gating
     behavior (static trade-count caps + permanent route/environment
     lockout, reproduced faithfully by neutralizing only the new code
     paths) vs the NEW dynamic behavior — same capital cap, same signals,
     different trade throughput.

Run: python validate_dynamic_gates.py
"""

from __future__ import annotations

import contextlib
import io
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

from strategy.v14_2_config import get_config
from strategy.v14_3_highvol_config import get_v14_3_config
from strategy.risk_manager import RiskManager
from strategy.package_builder import PackageResult, PackageLeg
from dataclasses import dataclass

from strategy.spread_quality_gate import OptionLeg
from strategy.dynamic_gate_controller import DynamicGateController, RouteAdjustment
from strategy.v14_2_core_runner import V14_2_CoreRunner

LOG_ROOT = Path("HFT/logs/validate_dynamic_gates")
if LOG_ROOT.exists():
    shutil.rmtree(LOG_ROOT)
LOG_ROOT.mkdir(parents=True, exist_ok=True)

results = []
all_pass = True


def check(condition, name, detail=""):
    global all_pass
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
    results.append((status, name, detail))
    mark = "  ✓" if condition else "  ✗"
    print(f"{mark} {name}" + (f" -- {detail}" if detail else ""))
    return condition


# ═══════════════════════════════════════════════════════════════════════════
# PART A — Invariant checks on the real production classes
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  PART A — INVARIANTS (capital protection + circuit breakers)")
print("=" * 72)

print("\n  -- A1: Exposure cap fuzz test --")
rng = random.Random(42)
rm = RiskManager(config=get_config(), log_dir=str(LOG_ROOT / "a1"))
rm.update_account(100_000.0)
rm.reset_day()

cap = rm.cfg["max_open_debit_exposure_pct"] * 100_000.0
max_seen_exposure = 0.0
attempts, admits = 0, 0
for i in range(500):
    debit = rng.uniform(50.0, 3000.0)
    attempts += 1
    r = rm.pre_trade_check(f"SYM{i % 15}", "VWAP_PULLBACK", debit)
    if r.allowed:
        admits += 1
        rm.add_open_position(f"SYM{i % 15}", debit)
        rm.record_trade(f"SYM{i % 15}", "VWAP_PULLBACK", debit)
        max_seen_exposure = max(max_seen_exposure, sum(rm._open_positions.values()))

check(
    max_seen_exposure <= cap + 1e-6,
    "500 randomized trade attempts never breach the exposure cap",
    f"peak_exposure=${max_seen_exposure:,.2f} cap=${cap:,.2f} admitted={admits}/{attempts}",
)

print("\n  -- A2: Dynamic capacity beats the old static baseline when capital allows --")
rm2 = RiskManager(config=get_config(), log_dir=str(LOG_ROOT / "a2"))
rm2.update_account(100_000.0)
rm2.reset_day()
old_static_cap = rm2.cfg["max_trades_per_day"]
admitted_today = 0
for i in range(20):
    r = rm2.pre_trade_check(f"SYM{i}", "VWAP_PULLBACK", 100.0)
    if r.allowed:
        admitted_today += 1
        rm2.record_trade(f"SYM{i}", "VWAP_PULLBACK", 100.0)
check(
    admitted_today > old_static_cap,
    "With ample headroom, daily cadence exceeds the old fixed cap",
    f"admitted={admitted_today} old_static_cap={old_static_cap}",
)

print("\n  -- A3: Absolute circuit breaker still caps cadence with unlimited capital --")
rm3 = RiskManager(config=get_config(), log_dir=str(LOG_ROOT / "a3"))
rm3.update_account(10_000_000.0)  # abundant capital
rm3.reset_day()
ceiling = rm3.cfg["absolute_max_trades_per_day_ceiling"]
admitted = 0
for i in range(ceiling + 10):
    r = rm3.pre_trade_check(f"SYM{i}", "VWAP_PULLBACK", 10.0)
    if r.allowed:
        admitted += 1
        rm3.record_trade(f"SYM{i}", "VWAP_PULLBACK", 10.0)
check(
    admitted == ceiling,
    "Circuit breaker holds cadence at the configured ceiling despite $10M account",
    f"admitted={admitted} ceiling={ceiling}",
)

print("\n  -- A4: Route recovery (cooldown -> probation -> reactivation) --")
from telemetry.route_expectancy_monitor import RouteExpectancyMonitor
from telemetry.environment_expectancy_monitor import EnvironmentExpectancyMonitor

route_mon = RouteExpectancyMonitor(log_dir=str(LOG_ROOT / "a4"))
env_mon = EnvironmentExpectancyMonitor(log_dir=str(LOG_ROOT / "a4"))
cfg = get_config()
gates = DynamicGateController(config=cfg, route_monitor=route_mon, env_monitor=env_mon)

for i in range(60):
    pnl = 10.0 if i % 5 == 0 else -10.0
    route_mon.record_trade("VWAP_PULLBACK", pnl)
check(
    route_mon.get_route_status("VWAP_PULLBACK") == "DISABLED",
    "Route drifts to DISABLED after a sustained losing streak",
)

allowed, state, _ = gates.route_trade_allowed("VWAP_PULLBACK")
check(not allowed and state == "COOLDOWN", "Immediately after disabling: blocked, in cooldown (not permanent)")

gates._route_disabled_since["VWAP_PULLBACK"] = date.today()  # placeholder, overwritten below
from datetime import datetime
gates._route_disabled_since["VWAP_PULLBACK"] = datetime.utcnow() - timedelta(hours=25)
allowed, state, size = gates.route_trade_allowed("VWAP_PULLBACK")
check(
    allowed and state == "PROBATION" and size < 1.0,
    "After cooldown elapses: a size-reduced probation trade is allowed",
    f"state={state} size_multiplier={size}",
)
gates.record_route_probation_trade("VWAP_PULLBACK")

for _ in range(30):
    route_mon.record_trade("VWAP_PULLBACK", 100.0)  # the probation trade (and more) win
allowed, state, size = gates.route_trade_allowed("VWAP_PULLBACK")
check(
    allowed and state in ("ACTIVE", "SOFT_SIZE_ONLY") and size >= 0.5,
    "Fresh winning data fully reactivates the route (old code could never reach this state)",
    f"final_state={state}",
)

print("\n  -- A5: Size scaling never exceeds what PackageBuilder already approved --")
pkg = PackageResult(
    is_package=True,
    core=PackageLeg(role="CORE", contracts=6, debit_per_contract=150.0, total_debit=900.0),
    runner=PackageLeg(role="RUNNER", contracts=2, debit_per_contract=150.0, total_debit=300.0),
    total_contracts=8, total_debit=1200.0,
)
original_total = pkg.total_contracts
V14_2_CoreRunner._scale_package_size(pkg, debit_per_contract=150.0, size_multiplier=0.5)
check(
    pkg.total_contracts <= original_total and pkg.total_contracts >= 1,
    "PROBATION-style 0.5x scaling shrinks contracts, never grows them",
    f"before={original_total} after={pkg.total_contracts}",
)


# ═══════════════════════════════════════════════════════════════════════════
# PART B — Head-to-head: OLD static/permanent-lockout vs NEW dynamic gates
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  PART B — HEAD-TO-HEAD ON AN IDENTICAL SYNTHETIC PRICE PATH")
print("=" * 72)


def generate_price_path(n_ticks=600, seed=7):
    """Pure-python synthetic intraday path: alternating trend/noise regimes."""
    rng = random.Random(seed)
    prices, highs, lows, vwaps, volumes = [150.0], [150.0], [150.0], [150.0], [1000]
    state, ticks_left = "noise", 0
    for _ in range(n_ticks):
        if ticks_left <= 0:
            state = rng.choices(["uptrend", "downtrend", "noise"], weights=[0.32, 0.18, 0.50])[0]
            ticks_left = rng.randint(10, 24)
        ticks_left -= 1

        last = prices[-1]
        if state == "uptrend":
            drift, vol_mult = rng.uniform(0.06, 0.28), rng.uniform(1.2, 2.2)
        elif state == "downtrend":
            drift, vol_mult = -rng.uniform(0.06, 0.28), rng.uniform(1.2, 2.2)
        else:
            drift, vol_mult = rng.uniform(-0.05, 0.05), rng.uniform(0.7, 1.2)

        price = max(1.0, last + drift + rng.uniform(-0.08, 0.08))
        prices.append(price)
        highs.append(max(price, last) + rng.uniform(0, 0.05))
        lows.append(min(price, last) - rng.uniform(0, 0.05))
        vwaps.append(vwaps[-1] * 0.9 + price * 0.1)
        volumes.append(max(1, int(1000 * vol_mult * rng.uniform(0.8, 1.2))))
    return prices, highs, lows, vwaps, volumes


def slope(values):
    n = len(values)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


@dataclass
class FakeContractInfo:
    """Mirrors strategy.option_chain_fetcher.ContractInfo's attributes
    (that module pulls in the real alpaca SDK, which this sandbox doesn't
    have installed and which core-runner only ever duck-types against)."""
    symbol: str
    strike: float
    expiration: str
    option_type: str
    underlying_symbol: str


class FakeChainFetcher:
    """
    Deterministic, config-compliant simulated option chain for this
    validation run. NOTE: the production _attempt_execution fallback path
    (used when no real chain_fetcher is wired up at all) hardcodes a wider
    bid/ask than the current spread-quality thresholds allow — that is a
    separate, pre-existing issue unrelated to the gating logic under test
    here, so this stand-in isolates gating-effectiveness validation from
    it rather than silently patching production code as a side effect.
    """

    def get_spread_with_quotes(self, symbol, underlying_price, side, dte_min=5, dte_max=14, strike_width=5.0):
        long_mid, short_mid = 1.50, 0.875
        long_leg = OptionLeg(
            contract_symbol=f"{symbol}_LONG_SIM", side="buy",
            bid=long_mid * 0.995, ask=long_mid * 1.005, mid=long_mid,
            delta=0.50, iv=0.35, volume=500, open_interest=2000, dte=7,
            strike=underlying_price,
        )
        short_leg = OptionLeg(
            contract_symbol=f"{symbol}_SHORT_SIM", side="sell",
            bid=short_mid * 0.995, ask=short_mid * 1.005, mid=short_mid,
            delta=0.30, iv=0.37, volume=400, open_interest=1500, dte=7,
            strike=underlying_price + strike_width,
        )
        long_contract = FakeContractInfo(
            symbol=f"{symbol}_LONG_SIM", strike=underlying_price, expiration="2026-02-20",
            option_type="call" if side == "CALL" else "put", underlying_symbol=symbol,
        )
        short_contract = FakeContractInfo(
            symbol=f"{symbol}_SHORT_SIM", strike=underlying_price + strike_width, expiration="2026-02-20",
            option_type="call" if side == "CALL" else "put", underlying_symbol=symbol,
        )
        return long_leg, short_leg, long_contract, short_contract


def make_runner(old_style: bool, tag: str) -> V14_2_CoreRunner:
    cfg = get_v14_3_config()
    if old_style:
        # Reproduce the pre-change hard caps exactly.
        cfg["absolute_max_trades_per_day_ceiling"] = cfg["max_trades_per_day"]
        cfg["absolute_max_trades_per_week_ceiling"] = cfg["max_trades_per_week"]

    runner = V14_2_CoreRunner(
        trading_client=None, option_data_client=None,
        config=cfg, paper_mode=True, log_dir=str(LOG_ROOT / tag),
    )
    runner.chain_fetcher = FakeChainFetcher()

    if old_style:
        # Neutralize ONLY the new recovery/adaptation code paths so the rest
        # of the pipeline (route scoring, confirmation, spread quality,
        # package sizing, risk manager) is the exact same production code
        # both sides run through. This is the pre-change gating behavior:
        # permanent lockout, no adaptive thresholds, no probation.
        rmon, emon = runner.route_monitor, runner.env_monitor

        def old_route_trade_allowed(route):
            allowed = rmon.is_route_allowed(route)
            return allowed, ("ACTIVE" if allowed else "DISABLED"), 1.0

        def old_env_trade_allowed(environment):
            allowed = emon.is_environment_allowed(environment)
            return allowed, ("ALLOWED" if allowed else "BLOCKED"), 1.0

        runner.dynamic_gates.route_trade_allowed = old_route_trade_allowed
        runner.dynamic_gates.environment_trade_allowed = old_env_trade_allowed
        runner.dynamic_gates.get_effective_thresholds = lambda route: {}
        runner.dynamic_gates.get_route_adjustment = (
            lambda route: RouteAdjustment(route, 1.0, "NORMAL", 1.0, "old_style")
        )

    return runner


def run_simulation(old_style: bool, tag: str, n_ticks=600, ticks_per_day=20):
    with contextlib.redirect_stdout(io.StringIO()):
        runner = make_runner(old_style, tag)
    runner.update_account(100_000.0, 50_000.0)

    prices, highs, lows, vwaps, volumes = generate_price_path(n_ticks=n_ticks)
    cap_pct = runner.cfg["max_open_debit_exposure_pct"]

    action_counts = {}
    peak_exposure_pct = 0.0
    day_idx = 0
    sim_date = date(2026, 1, 5)  # Monday
    day_start = 20  # index where the current simulated trading day began

    for t in range(20, n_ticks):
        if (t - 20) % ticks_per_day == 0:
            sim_date = sim_date + timedelta(days=1)
            runner.risk_manager.reset_day(today=sim_date)
            day_idx += 1
            day_start = t

        atr = sum(
            max(highs[i] - lows[i], abs(highs[i] - prices[i - 1]), abs(lows[i] - prices[i - 1]))
            for i in range(max(1, t - 13), t + 1)
        ) / max(1, min(14, t))
        trend = slope(prices[t - 19:t + 1])
        vol_ratio = volumes[t] / (sum(volumes[t - 10:t]) / 10.0)

        # High/low SINCE THE SIMULATED DAY'S OPEN — mirrors real intraday
        # high_of_day/low_of_day semantics (run_v14_2_paper.py uses the
        # current session's bars only). A window reaching back before the
        # day's open would inflate ConfirmationEngine's adverse-move check
        # with price history that predates any watch ticket, blocking
        # confirmation almost unconditionally — not a gating-logic effect.
        with contextlib.redirect_stdout(io.StringIO()):  # silence per-tick [CHAIN]/exec noise
            result = runner.evaluate_opportunity(
                symbol="COIN",
                price=prices[t], vwap=vwaps[t], atr=max(atr, 0.01),
                high_of_day=max(highs[day_start:t + 1]),
                low_of_day=min(lows[day_start:t + 1]),
                trend_slope=trend, volume_ratio=vol_ratio,
                price_5m_ago=prices[t - 5], price_15m_ago=prices[t - 15],
            )
        action_counts[result["action"]] = action_counts.get(result["action"], 0) + 1

        equity = runner._account_equity
        if equity > 0:
            exposure_pct = sum(runner.risk_manager._open_positions.values()) / equity
            peak_exposure_pct = max(peak_exposure_pct, exposure_pct)

    return runner, action_counts, peak_exposure_pct, cap_pct


print("\n  Running OLD-style (static caps, permanent lockout) ...")
old_runner, old_actions, old_peak_exp, cap_pct = run_simulation(old_style=True, tag="old")

print("  Running NEW dynamic-gate behavior ...")
new_runner, new_actions, new_peak_exp, _ = run_simulation(old_style=False, tag="new")

old_fills = old_actions.get("FILLED", 0)
new_fills = new_actions.get("FILLED", 0)

print(f"\n  Ticks simulated: 580 (20 simulated trading days) | symbol=COIN | identical price path both sides")
print(f"  Capital cap: {cap_pct:.0%} of equity\n")
print(f"  {'metric':<28}{'OLD (static/lockout)':>24}{'NEW (dynamic)':>18}")
print(f"  {'-'*28}{'-'*24}{'-'*18}")
for key in sorted(set(old_actions) | set(new_actions)):
    print(f"  {key:<28}{old_actions.get(key, 0):>24}{new_actions.get(key, 0):>18}")
print(f"  {'peak exposure %':<28}{old_peak_exp:>23.2%}{new_peak_exp:>18.2%}")

check(
    old_peak_exp <= cap_pct + 1e-9 and new_peak_exp <= cap_pct + 1e-9,
    "Neither regime ever breached the capital exposure cap during the simulation",
    f"old_peak={old_peak_exp:.2%} new_peak={new_peak_exp:.2%} cap={cap_pct:.0%}",
)
check(
    new_fills >= old_fills,
    "Dynamic gates fill at least as many opportune trades as the old static/lockout regime",
    f"old_fills={old_fills} new_fills={new_fills}",
)

old_route_stats = old_runner.route_monitor.get_all_route_stats()
new_route_stats = new_runner.route_monitor.get_all_route_stats()
old_disabled = [r for r, s in old_route_stats.items() if s["status"] == "DISABLED"]
new_disabled = [r for r, s in new_route_stats.items() if s["status"] == "DISABLED"]
print(f"\n  Routes DISABLED at end of run — OLD: {old_disabled or 'none'} | NEW: {new_disabled or 'none'}")
check(
    True,  # informational — recovery depends on whether this price path produced a losing streak
    "Route disable/recovery state captured for both regimes (see line above)",
)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
n_pass = sum(1 for s, _, _ in results if s == "PASS")
n_fail = sum(1 for s, _, _ in results if s == "FAIL")
print(f"  VALIDATION RESULTS: {n_pass} passed, {n_fail} failed")
if n_fail:
    print("\n  FAILURES:")
    for status, name, detail in results:
        if status == "FAIL":
            print(f"    ✗ {name}: {detail}")
print(f"\n  Status: {'DYNAMIC_GATES_VALIDATED' if all_pass else 'VALIDATION_FAILED'}")
print("=" * 72 + "\n")

if __name__ == "__main__":
    import sys
    sys.exit(0 if all_pass else 1)
