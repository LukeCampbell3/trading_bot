"""Timestamp-safe option-spread shadow labelling for V14.5.1.

This module turns observed, frozen verticals into independent option outcomes for
EmpiricalOptionEdgeMemory.  It never derives IC/EV from route score.  A label is
created only from quotes strictly AFTER the candidate timestamp, with entry at
the executable natural debit and exit at the executable natural credit.

The labeler is deliberately data-source agnostic: callers can populate
VerticalQuoteSnapshot from Alpaca historical quote data, OPRA capture, or paper
quote logs.  That keeps the causality/fill contract testable without credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import math

from strategy.empirical_option_edge import EmpiricalOptionEdgeMemory


@dataclass(frozen=True)
class ShadowCandidate:
    timestamp: datetime
    symbol: str
    route: str
    side: str
    route_score: float
    long_contract: str
    short_contract: str
    watch_spread_mid: float
    regime: str = "UNKNOWN"


@dataclass(frozen=True)
class VerticalQuoteSnapshot:
    timestamp: datetime
    symbol: str
    long_contract: str
    short_contract: str
    long_bid: float
    long_ask: float
    short_bid: float
    short_ask: float

    @property
    def entry_natural_debit(self) -> float:
        return max(0.0, float(self.long_ask) - float(self.short_bid))

    @property
    def exit_natural_credit(self) -> float:
        return max(0.0, float(self.long_bid) - float(self.short_ask))

    @property
    def spread_mid(self) -> float:
        long_mid = (float(self.long_bid) + float(self.long_ask)) / 2.0
        short_mid = (float(self.short_bid) + float(self.short_ask)) / 2.0
        return max(0.0, long_mid - short_mid)

    @property
    def valid(self) -> bool:
        return (
            self.long_bid > 0 and self.long_ask >= self.long_bid
            and self.short_bid > 0 and self.short_ask >= self.short_bid
            and self.entry_natural_debit > 0
        )


@dataclass
class ShadowLabel:
    candidate_timestamp: datetime
    entry_timestamp: Optional[datetime]
    exit_timestamp: Optional[datetime]
    symbol: str
    route: str
    side: str
    route_score: float
    long_contract: str
    short_contract: str
    regime: str
    horizon_minutes: int
    intended_limit: float
    entry_debit: float
    exit_credit: float
    spread_return: float
    filled: bool
    label_valid: bool
    missed_fill_reason: str = ""


class OptionSpreadShadowLabeler:
    """Generate causal, executable-quote labels for frozen debit verticals."""

    def __init__(
        self,
        *,
        max_limit_chase_pct: float = 0.034,
        max_entry_delay_seconds: float = 120.0,
        max_exit_delay_seconds: float = 120.0,
        primary_horizon_minutes: int = 60,
        horizons_minutes: Sequence[int] = (30, 60, 120),
        output_dir: str = "HFT/logs/v14_5_1/shadow_labels",
    ):
        self.max_limit_chase_pct = float(max_limit_chase_pct)
        self.max_entry_delay_seconds = float(max_entry_delay_seconds)
        self.max_exit_delay_seconds = float(max_exit_delay_seconds)
        self.primary_horizon_minutes = int(primary_horizon_minutes)
        self.horizons_minutes = tuple(sorted(set(int(x) for x in horizons_minutes if int(x) > 0)))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _key(symbol: str, long_contract: str, short_contract: str) -> Tuple[str, str, str]:
        return str(symbol).upper(), str(long_contract), str(short_contract)

    def _index_quotes(self, quotes: Iterable[VerticalQuoteSnapshot]):
        indexed: Dict[Tuple[str, str, str], List[VerticalQuoteSnapshot]] = {}
        for q in quotes:
            indexed.setdefault(self._key(q.symbol, q.long_contract, q.short_contract), []).append(q)
        for key in indexed:
            indexed[key].sort(key=lambda x: self._utc(x.timestamp))
        return indexed

    def _first_quote_after(
        self,
        quotes: List[VerticalQuoteSnapshot],
        after: datetime,
        max_delay_seconds: float,
    ) -> Optional[VerticalQuoteSnapshot]:
        after = self._utc(after)
        for quote in quotes:
            ts = self._utc(quote.timestamp)
            if ts <= after:  # strict causality: never use same/earlier timestamp
                continue
            delay = (ts - after).total_seconds()
            if delay > max_delay_seconds:
                return None
            if quote.valid:
                return quote
        return None

    def _quote_at_or_after_target(
        self,
        quotes: List[VerticalQuoteSnapshot],
        target: datetime,
        max_delay_seconds: float,
    ) -> Optional[VerticalQuoteSnapshot]:
        target = self._utc(target)
        for quote in quotes:
            ts = self._utc(quote.timestamp)
            if ts < target:
                continue
            if (ts - target).total_seconds() > max_delay_seconds:
                return None
            if quote.valid and quote.exit_natural_credit > 0:
                return quote
        return None

    def label_candidate(
        self,
        candidate: ShadowCandidate,
        quotes: Iterable[VerticalQuoteSnapshot],
    ) -> List[ShadowLabel]:
        series = sorted(list(quotes), key=lambda q: self._utc(q.timestamp))
        watch_mid = float(candidate.watch_spread_mid)
        intended_limit = watch_mid * (1.0 + self.max_limit_chase_pct) if watch_mid > 0 else 0.0
        entry = self._first_quote_after(
            series, candidate.timestamp, self.max_entry_delay_seconds
        )
        if entry is None:
            return [self._miss(candidate, h, intended_limit, "no_fresh_entry_quote") for h in self.horizons_minutes]
        if intended_limit <= 0:
            return [self._miss(candidate, h, intended_limit, "invalid_watch_mid") for h in self.horizons_minutes]
        if entry.entry_natural_debit > intended_limit + 1e-9:
            return [self._miss(candidate, h, intended_limit, "natural_ask_above_limit", entry.timestamp) for h in self.horizons_minutes]

        # Conservative fill model: if the natural ask is marketable at our limit,
        # pay the natural debit.  We do not award midpoint improvement.
        entry_debit = entry.entry_natural_debit
        out = []
        for horizon in self.horizons_minutes:
            target = self._utc(entry.timestamp) + timedelta(minutes=horizon)
            exit_q = self._quote_at_or_after_target(
                series, target, self.max_exit_delay_seconds
            )
            if exit_q is None:
                out.append(self._miss(
                    candidate, horizon, intended_limit, "no_fresh_exit_quote", entry.timestamp,
                    filled=True, entry_debit=entry_debit,
                ))
                continue
            credit = exit_q.exit_natural_credit
            ret = (credit - entry_debit) / entry_debit if entry_debit > 0 else 0.0
            out.append(ShadowLabel(
                candidate_timestamp=self._utc(candidate.timestamp),
                entry_timestamp=self._utc(entry.timestamp),
                exit_timestamp=self._utc(exit_q.timestamp),
                symbol=candidate.symbol.upper(),
                route=candidate.route,
                side=candidate.side.upper(),
                route_score=float(candidate.route_score),
                long_contract=candidate.long_contract,
                short_contract=candidate.short_contract,
                regime=candidate.regime,
                horizon_minutes=horizon,
                intended_limit=intended_limit,
                entry_debit=entry_debit,
                exit_credit=credit,
                spread_return=ret,
                filled=True,
                label_valid=True,
            ))
        return out

    def _miss(
        self,
        c: ShadowCandidate,
        horizon: int,
        intended_limit: float,
        reason: str,
        entry_timestamp: Optional[datetime] = None,
        *,
        filled: bool = False,
        entry_debit: float = 0.0,
    ) -> ShadowLabel:
        return ShadowLabel(
            candidate_timestamp=self._utc(c.timestamp),
            entry_timestamp=self._utc(entry_timestamp) if entry_timestamp else None,
            exit_timestamp=None,
            symbol=c.symbol.upper(), route=c.route, side=c.side.upper(),
            route_score=float(c.route_score), long_contract=c.long_contract,
            short_contract=c.short_contract, regime=c.regime,
            horizon_minutes=int(horizon), intended_limit=float(intended_limit),
            entry_debit=float(entry_debit), exit_credit=0.0, spread_return=0.0,
            filled=bool(filled), label_valid=False, missed_fill_reason=reason,
        )

    def label_many(
        self,
        candidates: Iterable[ShadowCandidate],
        quotes: Iterable[VerticalQuoteSnapshot],
    ) -> List[ShadowLabel]:
        indexed = self._index_quotes(quotes)
        labels: List[ShadowLabel] = []
        for c in candidates:
            series = indexed.get(self._key(c.symbol, c.long_contract, c.short_contract), [])
            labels.extend(self.label_candidate(c, series))
        return labels

    def feed_primary_labels(
        self,
        labels: Iterable[ShadowLabel],
        edge_memory: EmpiricalOptionEdgeMemory,
        source: str = "SHADOW_REPLAY",
    ) -> int:
        count = 0
        for label in labels:
            if not label.label_valid or label.horizon_minutes != self.primary_horizon_minutes:
                continue
            ok = edge_memory.record(
                symbol=label.symbol,
                route=label.route,
                route_score=label.route_score,
                spread_return=label.spread_return,
                regime=label.regime,
                source=source,
                timestamp=label.entry_timestamp,
            )
            count += int(ok)
        return count

    def write_labels(self, labels: Iterable[ShadowLabel], filename: str = "v14_5_1_shadow_labels.csv") -> Path:
        path = self.output_dir / filename
        rows = list(labels)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "candidate_timestamp", "entry_timestamp", "exit_timestamp",
                "symbol", "route", "side", "route_score", "long_contract",
                "short_contract", "regime", "horizon_minutes", "intended_limit",
                "entry_debit", "exit_credit", "spread_return", "filled",
                "label_valid", "missed_fill_reason",
            ])
            for x in rows:
                w.writerow([
                    x.candidate_timestamp.isoformat(),
                    x.entry_timestamp.isoformat() if x.entry_timestamp else "",
                    x.exit_timestamp.isoformat() if x.exit_timestamp else "",
                    x.symbol, x.route, x.side, f"{x.route_score:.8f}",
                    x.long_contract, x.short_contract, x.regime, x.horizon_minutes,
                    f"{x.intended_limit:.6f}", f"{x.entry_debit:.6f}",
                    f"{x.exit_credit:.6f}", f"{x.spread_return:.8f}",
                    int(x.filled), int(x.label_valid), x.missed_fill_reason,
                ])
        return path
