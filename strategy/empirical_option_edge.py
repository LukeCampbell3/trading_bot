"""Independent option-edge memory for V14.5.1.

The old route conditioner derived IC and EV mechanically from route_score.  That
made three apparent gates one correlated gate.  This module keeps score and
future option-spread outcomes separate and computes rolling empirical evidence
from timestamp-safe labels.

The memory accepts both executed fills and replay/shadow labels.  It never
manufactures an IC when there is insufficient variation or sample size.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Tuple
import csv
import math

import numpy as np


@dataclass
class EdgeObservation:
    timestamp: str
    symbol: str
    route: str
    regime: str
    route_score: float
    spread_return: float
    source: str


@dataclass
class EdgeStats:
    samples: int = 0
    ready: bool = False
    ic_spread: float = 0.0
    ev_over_debit: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class EmpiricalOptionEdgeMemory:
    """Rolling route x symbol x regime option outcome memory."""

    def __init__(
        self,
        min_samples: int = 20,
        window: int = 200,
        prior_strength: float = 20.0,
        log_path: str = "HFT/logs/v14_5_1/empirical_option_edge.csv",
    ):
        self.min_samples = int(min_samples)
        self.window = int(window)
        self.prior_strength = float(prior_strength)
        self._data: Dict[Tuple[str, str, str], Deque[EdgeObservation]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "symbol", "route", "regime", "route_score",
                    "spread_return", "source",
                ])

    @staticmethod
    def _key(symbol: str, route: str, regime: str) -> Tuple[str, str, str]:
        return (str(symbol).upper(), str(route), str(regime or "UNKNOWN"))

    def record(
        self,
        *,
        symbol: str,
        route: str,
        route_score: float,
        spread_return: float,
        regime: str = "UNKNOWN",
        source: str = "SHADOW",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        score = float(route_score)
        ret = float(spread_return)
        if not (math.isfinite(score) and math.isfinite(ret)):
            return False
        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        obs = EdgeObservation(
            timestamp=ts.astimezone(timezone.utc).isoformat(),
            symbol=str(symbol).upper(),
            route=str(route),
            regime=str(regime or "UNKNOWN"),
            route_score=score,
            spread_return=ret,
            source=str(source),
        )
        self._data[self._key(symbol, route, regime)].append(obs)
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                obs.timestamp, obs.symbol, obs.route, obs.regime,
                f"{obs.route_score:.8f}", f"{obs.spread_return:.8f}", obs.source,
            ])
        return True

    def _observations(self, symbol: str, route: str, regime: str) -> list[EdgeObservation]:
        exact = list(self._data.get(self._key(symbol, route, regime), ()))
        if len(exact) >= self.min_samples:
            return exact
        # Before exact regime evidence matures, pool the same symbol + route across
        # regimes.  This is still independent realized evidence, not a score proxy.
        pooled = []
        symbol = str(symbol).upper()
        route = str(route)
        for (s, r, _), values in self._data.items():
            if s == symbol and r == route:
                pooled.extend(values)
        pooled.sort(key=lambda x: x.timestamp)
        return pooled[-self.window:]

    def stats(self, symbol: str, route: str, regime: str = "UNKNOWN") -> EdgeStats:
        obs = self._observations(symbol, route, regime)
        n = len(obs)
        if n == 0:
            return EdgeStats()
        scores = np.asarray([x.route_score for x in obs], dtype=float)
        rets = np.asarray([x.spread_return for x in obs], dtype=float)
        wins = rets > 0
        pos = float(rets[rets > 0].sum())
        neg = abs(float(rets[rets < 0].sum()))
        pf = pos / neg if neg > 1e-12 else (float("inf") if pos > 0 else 0.0)

        ic = 0.0
        if n >= 3 and float(np.std(scores)) > 1e-9 and float(np.std(rets)) > 1e-9:
            value = float(np.corrcoef(scores, rets)[0, 1])
            if math.isfinite(value):
                ic = value

        # Shrink both estimates toward zero until the evidence base is substantial.
        shrink = n / (n + self.prior_strength) if self.prior_strength > 0 else 1.0
        return EdgeStats(
            samples=n,
            ready=n >= self.min_samples,
            ic_spread=ic * shrink,
            ev_over_debit=float(np.mean(rets)) * shrink,
            win_rate=float(np.mean(wins)),
            profit_factor=pf,
        )

    def annotate_candidate(self, candidate, symbol: str, regime: str = "UNKNOWN"):
        st = self.stats(symbol, candidate.route, regime)
        # Deliberately overwrite the old synthetic values.  Callers can inspect the
        # empirical_* fields to decide whether the gate is mature.
        candidate.ic_spread = st.ic_spread if st.ready else 0.0
        candidate.ev_over_debit = st.ev_over_debit if st.ready else 0.0
        candidate.empirical_edge_ready = st.ready
        candidate.empirical_edge_samples = st.samples
        candidate.empirical_ic_spread = st.ic_spread
        candidate.empirical_ev_over_debit = st.ev_over_debit
        candidate.empirical_win_rate = st.win_rate
        candidate.empirical_profit_factor = st.profit_factor
        return candidate

    def load_csv(self, path: str) -> int:
        """Load timestamp-safe replay/shadow labels exported by the validator."""
        p = Path(path)
        if not p.exists():
            return 0
        count = 0
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
                    ok = self.record(
                        symbol=row["symbol"], route=row["route"],
                        route_score=float(row["route_score"]),
                        spread_return=float(row["spread_return"]),
                        regime=row.get("regime", "UNKNOWN"),
                        source=row.get("source", "REPLAY_IMPORT"), timestamp=ts,
                    )
                    count += int(ok)
                except Exception:
                    continue
        return count
