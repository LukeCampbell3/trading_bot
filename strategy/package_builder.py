"""
Package Builder for V14.2 Core Runner

Builds core-runner packages when conditions qualify.
Falls back to V12.2 single-spread mode when they don't.

Core: high-probability early profit capture (core_fraction of debit).
Runner: continuation upside (runner_fraction of debit), locked after core pays.

IMPORTANT:
- If account cannot afford two separate spreads, use fallback mode.
- Never fake fills. Never over-size to force a package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG
from strategy.spread_quality_gate import SpreadQualityReport


@dataclass
class PackageLeg:
    """Logical leg of a core-runner package."""
    role: str = ""  # "CORE" or "RUNNER"
    contracts: int = 0
    debit_per_contract: float = 0.0
    total_debit: float = 0.0
    target_pct: float = 0.0
    stop_pct: float = 0.0
    lock_pct: float = 0.0  # runner only: raised after core pays


@dataclass
class PackageResult:
    """Result of the package builder."""
    is_package: bool = False
    is_fallback: bool = False
    fallback_reason: str = ""

    # Package legs
    core: Optional[PackageLeg] = None
    runner: Optional[PackageLeg] = None

    # Single spread fallback
    single_contracts: int = 0
    single_debit: float = 0.0
    single_target_pct: float = 0.0
    single_stop_pct: float = 0.0

    # Totals
    total_debit: float = 0.0
    total_contracts: int = 0

    @property
    def fractions_valid(self) -> bool:
        """Verify core + runner fractions sum to ~1.0."""
        if not self.is_package:
            return True
        if self.core and self.runner:
            total = (self.core.total_debit + self.runner.total_debit)
            if total > 0:
                core_frac = self.core.total_debit / total
                runner_frac = self.runner.total_debit / total
                return abs((core_frac + runner_frac) - 1.0) < 0.01
        return False


class PackageBuilder:
    """
    Determines whether to build a core-runner package or fall back to V12.2 single spread.
    
    Package requires:
    - package_min_q quality score
    - package_min_pwin probability
    - Acceptable environment stress
    - Route is package-eligible (not soft-only)
    - Account can afford at least 2 contracts
    - Spread debit within limits
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or CFG

    def build(
        self,
        quality_report: SpreadQualityReport,
        quality_score: float,
        pwin: float,
        env_stress: float,
        route: str,
        account_buying_power: float,
        current_iv: float = 0.0,
    ) -> PackageResult:
        """
        Attempt to build a core-runner package.
        
        Parameters:
            quality_report: SpreadQualityReport from the spread quality gate
            quality_score: Composite quality score [0,1]
            pwin: Estimated probability of winning
            env_stress: Current environment stress [0,1]
            route: Route name
            account_buying_power: Available buying power in dollars
            current_iv: Current implied volatility
        """
        result = PackageResult()
        spread_mid = quality_report.spread_mid
        debit_per_contract = spread_mid * 100  # per-contract cost in dollars

        if debit_per_contract <= 0:
            result.is_fallback = True
            result.fallback_reason = "zero_debit"
            return self._build_fallback(result, 0, account_buying_power)

        # ─── Package Qualification Checks ────────────────────────────────
        # Check 1: Quality score
        q_threshold = self.cfg["package_min_q"]
        if current_iv > 0.5:  # High IV regime
            q_threshold = self.cfg["hi_iv_min_q"]

        if quality_score < q_threshold:
            result.is_fallback = True
            result.fallback_reason = f"quality_score_too_low: {quality_score:.4f} < {q_threshold}"
            return self._build_fallback(result, debit_per_contract, account_buying_power)

        # Check 2: Win probability
        if pwin < self.cfg["package_min_pwin"]:
            result.is_fallback = True
            result.fallback_reason = f"pwin_too_low: {pwin:.4f} < {self.cfg['package_min_pwin']}"
            return self._build_fallback(result, debit_per_contract, account_buying_power)

        # Check 3: Environment stress
        if env_stress > self.cfg["max_env_stress"]:
            result.is_fallback = True
            result.fallback_reason = f"env_stress: {env_stress:.4f} > {self.cfg['max_env_stress']}"
            return self._build_fallback(result, debit_per_contract, account_buying_power)

        # Check 4: Route must be package-eligible (not soft-only)
        if route not in self.cfg["routes_allowed"]:
            result.is_fallback = True
            result.fallback_reason = f"route_not_package_eligible: {route}"
            return self._build_fallback(result, debit_per_contract, account_buying_power)

        # Check 5: Debit within limits
        scaled_debit = debit_per_contract * self.cfg["package_debit_scale"]
        if scaled_debit > self.cfg["max_package_debit"]:
            result.is_fallback = True
            result.fallback_reason = (
                f"scaled_debit_exceeds_max: {scaled_debit:.2f} > {self.cfg['max_package_debit']}"
            )
            return self._build_fallback(result, debit_per_contract, account_buying_power)

        # Check 6: Account can afford at least 2 contracts
        min_package_cost = debit_per_contract * 2
        if account_buying_power < min_package_cost:
            result.is_fallback = True
            result.fallback_reason = (
                f"insufficient_buying_power: {account_buying_power:.2f} < {min_package_cost:.2f}"
            )
            return self._build_fallback(result, debit_per_contract, account_buying_power)

        # ─── Build Package ───────────────────────────────────────────────
        return self._build_package(result, debit_per_contract, account_buying_power)

    def _build_package(
        self, result: PackageResult, debit_per_contract: float, buying_power: float
    ) -> PackageResult:
        """Build core-runner package with proper fractions."""
        # Determine total contracts we can afford
        max_exposure = buying_power * self.cfg["max_open_debit_exposure_pct"]
        max_contracts = int(max_exposure / debit_per_contract) if debit_per_contract > 0 else 0
        max_contracts = max(2, min(max_contracts, 10))  # at least 2, cap at 10

        # Split into core and runner by fraction
        core_contracts = max(1, round(max_contracts * self.cfg["core_fraction"]))
        runner_contracts = max(1, max_contracts - core_contracts)

        # Ensure we can actually afford this
        total_cost = (core_contracts + runner_contracts) * debit_per_contract
        if total_cost > buying_power * self.cfg["max_open_debit_exposure_pct"]:
            # Scale down
            runner_contracts = max(1, runner_contracts - 1)
            if (core_contracts + runner_contracts) * debit_per_contract > buying_power * self.cfg["max_open_debit_exposure_pct"]:
                core_contracts = max(1, core_contracts - 1)

        result.is_package = True
        result.core = PackageLeg(
            role="CORE",
            contracts=core_contracts,
            debit_per_contract=debit_per_contract,
            total_debit=core_contracts * debit_per_contract,
            target_pct=self.cfg["core_target_pct"],
            stop_pct=self.cfg["package_stop_pct"],
        )
        result.runner = PackageLeg(
            role="RUNNER",
            contracts=runner_contracts,
            debit_per_contract=debit_per_contract,
            total_debit=runner_contracts * debit_per_contract,
            target_pct=self.cfg["runner_target_pct"],
            stop_pct=self.cfg["package_stop_pct"],
            lock_pct=self.cfg["runner_lock_pct"],
        )
        result.total_debit = result.core.total_debit + result.runner.total_debit
        result.total_contracts = core_contracts + runner_contracts
        return result

    def _build_fallback(
        self, result: PackageResult, debit_per_contract: float, buying_power: float
    ) -> PackageResult:
        """Build V12.2 single-spread fallback."""
        result.is_fallback = True
        result.is_package = False

        if debit_per_contract <= 0:
            result.single_contracts = 0
            result.total_debit = 0.0
            return result

        max_exposure = buying_power * self.cfg["max_open_debit_exposure_pct"]
        max_contracts = int(max_exposure / debit_per_contract) if debit_per_contract > 0 else 0
        max_contracts = max(0, min(max_contracts, 5))

        if max_contracts == 0:
            result.fallback_reason += " | cannot_afford_single_spread"
            result.single_contracts = 0
            result.total_debit = 0.0
            return result

        result.single_contracts = max_contracts
        result.single_debit = max_contracts * debit_per_contract
        result.single_target_pct = self.cfg["soft_greed_target_pct"]
        result.single_stop_pct = self.cfg["single_spread_initial_stop_pct"]
        result.total_debit = result.single_debit
        result.total_contracts = max_contracts
        return result
