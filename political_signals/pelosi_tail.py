"""Nancy Pelosi disclosure tracker and stock-tail policy.

The tracker reacts to *public disclosure time*, never the original transaction date.
Congressional disclosures are delayed by law, so the policy explicitly measures that
lag and refuses to pretend that a later filing was known on the trade date.

Primary source: Quiver Quantitative live congressional trading API.
Authoritative verification can be layered on separately with House Clerk filings.

This module intentionally does not submit broker orders.  It produces deterministic,
auditable tail decisions that a paper/live execution layer may consume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


PELOSI_NAMES = ("nancy pelosi", "pelosi, nancy")
DEFAULT_QUIVER_URL = "https://api.quiverquant.com/beta/live/congresstrading"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(value: Any) -> Optional[date]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for candidate in (text, text[:10]):
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date()
        except Exception:
            pass
    return None


def _normal_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _parse_money_range(text: str) -> Tuple[float, float]:
    """Parse congressional amount ranges such as '$15,001 - $50,000'."""
    if not text:
        return 0.0, 0.0
    nums = []
    for raw in re.findall(r"\$?([0-9][0-9,]*(?:\.[0-9]+)?)", str(text)):
        try:
            nums.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    if not nums:
        return 0.0, 0.0
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums[0], nums[1]), max(nums[0], nums[1])


def _pick(raw: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return default


@dataclass(frozen=True)
class PelosiDisclosure:
    representative: str
    report_date: Optional[date]
    transaction_date: Optional[date]
    ticker: str
    transaction: str
    amount_range: str
    ticker_type: str
    description: str
    house: str = "House"
    source: str = "QUIVER"

    @classmethod
    def from_quiver(cls, raw: Dict[str, Any]) -> "PelosiDisclosure":
        return cls(
            representative=_normal_text(_pick(raw, "Representative", "Name", "Politician")),
            report_date=_parse_date(_pick(raw, "ReportDate", "Filed", "fileDate", "Date")),
            transaction_date=_parse_date(_pick(raw, "TransactionDate", "Traded", "TradeDate")),
            ticker=_normal_text(_pick(raw, "Ticker", "ticker")).upper(),
            transaction=_normal_text(_pick(raw, "Transaction", "TransactionType", "Type")),
            amount_range=_normal_text(_pick(raw, "Range", "AmountRange", "Amount")),
            ticker_type=_normal_text(_pick(raw, "TickerType", "AssetType", "TypeOfSecurity")),
            description=_normal_text(_pick(raw, "Description", "AssetDescription", "Notes")),
            house=_normal_text(_pick(raw, "House", "Chamber", default="House")),
        )

    @property
    def is_pelosi(self) -> bool:
        name = self.representative.lower()
        return any(x in name for x in PELOSI_NAMES)

    @property
    def disclosure_lag_days(self) -> Optional[int]:
        if not self.report_date or not self.transaction_date:
            return None
        return max(0, (self.report_date - self.transaction_date).days)

    @property
    def amount_bounds(self) -> Tuple[float, float]:
        return _parse_money_range(self.amount_range)

    @property
    def fingerprint(self) -> str:
        # Quiver payloads do not always expose a stable transaction id, so derive one
        # from the immutable disclosure fields that matter to this strategy.
        payload = "|".join([
            self.representative.lower(),
            self.report_date.isoformat() if self.report_date else "",
            self.transaction_date.isoformat() if self.transaction_date else "",
            self.ticker,
            self.transaction.lower(),
            self.amount_range,
            self.ticker_type.lower(),
            self.description.lower(),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @property
    def disclosed_direction(self) -> str:
        """Return BULLISH, BEARISH, or UNKNOWN from the filing text.

        We follow the underlying stock rather than reproducing Congressional option
        structures.  Purchased calls are bullish; purchased puts are bearish.
        Ambiguous options are never auto-classified.
        """
        tx = self.transaction.lower()
        detail = f"{self.ticker_type} {self.description}".lower()
        purchase = any(x in tx for x in ("purchase", "buy", "acquisition"))
        sale = any(x in tx for x in ("sale", "sell", "disposition"))
        is_put = bool(re.search(r"\bput\b", detail))
        is_call = bool(re.search(r"\bcall\b", detail))
        is_option = "option" in detail or is_put or is_call

        if is_option:
            if purchase and is_call:
                return "BULLISH"
            if purchase and is_put:
                return "BEARISH"
            if sale and is_call:
                return "BEARISH"
            if sale and is_put:
                return "BULLISH"
            return "UNKNOWN"
        if purchase:
            return "BULLISH"
        if sale:
            return "BEARISH"
        return "UNKNOWN"


@dataclass
class PelosiTailDecision:
    disclosure_id: str
    ticker: str
    action: str  # BUY, EXIT_ONLY, WATCH, IGNORE
    score: float
    target_notional_pct: float
    disclosure_lag_days: Optional[int]
    amount_low: float
    amount_high: float
    first_seen_at: str
    reasons: List[str]
    estimated_trade_to_now_return: Optional[float] = None

    @property
    def eligible(self) -> bool:
        return self.action in {"BUY", "EXIT_ONLY"}


class PelosiTailPolicy:
    """Turn a new Pelosi disclosure into a conservative stock-tail decision.

    The goal is to react quickly after public disclosure while avoiding the classic
    mistake of treating a 20-40 day old transaction as a fresh signal.
    """

    def __init__(
        self,
        *,
        max_disclosure_lag_days: int = 45,
        max_buy_runup_since_trade: float = 0.25,
        max_buy_drawdown_since_trade: float = 0.35,
        base_max_notional_pct: float = 0.08,
        min_buy_score: float = 0.42,
    ):
        self.max_disclosure_lag_days = int(max_disclosure_lag_days)
        self.max_buy_runup_since_trade = float(max_buy_runup_since_trade)
        self.max_buy_drawdown_since_trade = float(max_buy_drawdown_since_trade)
        self.base_max_notional_pct = float(base_max_notional_pct)
        self.min_buy_score = float(min_buy_score)

    @staticmethod
    def _lag_factor(days: Optional[int]) -> float:
        if days is None:
            return 0.45
        if days <= 7:
            return 1.00
        if days <= 14:
            return 0.85
        if days <= 30:
            return 0.65
        if days <= 45:
            return 0.45
        return 0.0

    @staticmethod
    def _amount_factor(low: float, high: float) -> float:
        midpoint = (low + high) / 2.0 if high > 0 else low
        if midpoint >= 1_000_000:
            return 1.00
        if midpoint >= 250_000:
            return 0.90
        if midpoint >= 50_000:
            return 0.78
        if midpoint >= 15_000:
            return 0.65
        if midpoint > 0:
            return 0.50
        return 0.45

    @staticmethod
    def _instrument_factor(d: PelosiDisclosure) -> float:
        detail = f"{d.ticker_type} {d.description}".lower()
        if "option" in detail or "call" in detail or "put" in detail:
            return 0.85
        return 1.00

    def decide(
        self,
        d: PelosiDisclosure,
        *,
        first_seen_at: Optional[datetime] = None,
        trade_to_now_return: Optional[float] = None,
    ) -> PelosiTailDecision:
        seen = first_seen_at or _utc_now()
        reasons: List[str] = []
        low, high = d.amount_bounds
        lag = d.disclosure_lag_days
        direction = d.disclosed_direction

        if not d.ticker or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", d.ticker):
            return self._decision(d, "IGNORE", 0.0, 0.0, seen, low, high, reasons + ["invalid_or_missing_ticker"], trade_to_now_return)
        if lag is not None and lag > self.max_disclosure_lag_days:
            return self._decision(d, "IGNORE", 0.0, 0.0, seen, low, high, reasons + ["disclosure_too_old"], trade_to_now_return)
        if direction == "UNKNOWN":
            return self._decision(d, "WATCH", 0.0, 0.0, seen, low, high, reasons + ["ambiguous_transaction_direction"], trade_to_now_return)

        lag_factor = self._lag_factor(lag)
        amount_factor = self._amount_factor(low, high)
        instrument_factor = self._instrument_factor(d)
        score = lag_factor * amount_factor * instrument_factor
        reasons.extend([
            f"lag_factor={lag_factor:.2f}",
            f"amount_factor={amount_factor:.2f}",
            f"instrument_factor={instrument_factor:.2f}",
        ])

        # Residual-opportunity gate.  A delayed disclosure that has already run hard
        # is not tailed at full speed.  Large adverse drift is also treated as a
        # changed thesis rather than an invitation to average down.
        if direction == "BULLISH" and trade_to_now_return is not None:
            if trade_to_now_return > self.max_buy_runup_since_trade:
                return self._decision(d, "WATCH", score * 0.50, 0.0, seen, low, high,
                                      reasons + ["positive_move_already_consumed"], trade_to_now_return)
            if trade_to_now_return < -self.max_buy_drawdown_since_trade:
                return self._decision(d, "WATCH", score * 0.50, 0.0, seen, low, high,
                                      reasons + ["large_adverse_move_changed_thesis"], trade_to_now_return)
            residual = max(0.55, 1.0 - max(0.0, trade_to_now_return) / max(self.max_buy_runup_since_trade, 1e-9))
            score *= residual
            reasons.append(f"residual_opportunity_factor={residual:.2f}")

        if direction == "BEARISH":
            # Stock mode is deliberately long-only for the Pelosi overlay.  A sale or
            # bearish option disclosure may exit a Pelosi-tail position but will not
            # create a naked short position.
            return self._decision(d, "EXIT_ONLY", score, 0.0, seen, low, high,
                                  reasons + ["bearish_disclosure_long_only_exit"], trade_to_now_return)

        if score < self.min_buy_score:
            return self._decision(d, "WATCH", score, 0.0, seen, low, high,
                                  reasons + ["tail_score_below_buy_floor"], trade_to_now_return)

        notional_pct = min(self.base_max_notional_pct, self.base_max_notional_pct * score)
        return self._decision(d, "BUY", score, notional_pct, seen, low, high,
                              reasons + ["fresh_public_disclosure_tail"], trade_to_now_return)

    @staticmethod
    def _decision(
        d: PelosiDisclosure,
        action: str,
        score: float,
        pct: float,
        seen: datetime,
        low: float,
        high: float,
        reasons: List[str],
        trade_to_now_return: Optional[float],
    ) -> PelosiTailDecision:
        return PelosiTailDecision(
            disclosure_id=d.fingerprint,
            ticker=d.ticker,
            action=action,
            score=float(max(0.0, min(1.0, score))),
            target_notional_pct=float(max(0.0, pct)),
            disclosure_lag_days=d.disclosure_lag_days,
            amount_low=low,
            amount_high=high,
            first_seen_at=seen.astimezone(timezone.utc).isoformat(),
            reasons=list(reasons),
            estimated_trade_to_now_return=trade_to_now_return,
        )


class QuiverCongressClient:
    """Minimal Quiver live Congress client with tolerant response normalization."""

    def __init__(self, api_key: Optional[str] = None, url: Optional[str] = None, timeout: float = 12.0):
        self.api_key = api_key or os.getenv("QUIVER_API_KEY", "")
        self.url = url or os.getenv("QUIVER_CONGRESS_URL", DEFAULT_QUIVER_URL)
        self.timeout = float(timeout)
        if not self.api_key:
            raise ValueError("QUIVER_API_KEY is required for the Pelosi tracker")

    def fetch_recent(self) -> List[PelosiDisclosure]:
        response = requests.get(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "trading_bot-pelosi-tail/1.0",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            rows = payload.get("data", payload.get("results", payload.get("trades", [])))
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("Unexpected Quiver Congress response shape")
        disclosures = [PelosiDisclosure.from_quiver(x) for x in rows if isinstance(x, dict)]
        return [x for x in disclosures if x.is_pelosi]


class PelosiTailPoller:
    """Deduplicating disclosure poller with restart-safe state and audit logs."""

    def __init__(
        self,
        client: QuiverCongressClient,
        policy: Optional[PelosiTailPolicy] = None,
        *,
        state_path: str = "HFT/logs/pelosi_tail/state.json",
        audit_path: str = "HFT/logs/pelosi_tail/disclosures.jsonl",
        seed_existing_on_first_run: bool = True,
    ):
        self.client = client
        self.policy = policy or PelosiTailPolicy()
        self.state_path = Path(state_path)
        self.audit_path = Path(audit_path)
        self.seed_existing_on_first_run = bool(seed_existing_on_first_run)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"initialized": False, "seen": [], "tail_positions": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            data.setdefault("initialized", True)
            data.setdefault("seen", [])
            data.setdefault("tail_positions", {})
            return data
        except Exception:
            # Corrupt state must fail closed: do not auto-trade a backlog.
            return {"initialized": False, "seen": [], "tail_positions": {}, "state_recovered": True}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def _audit(self, d: PelosiDisclosure, decision: Optional[PelosiTailDecision], event: str) -> None:
        record = {
            "event": event,
            "observed_at": _utc_now().isoformat(),
            "disclosure": {
                **asdict(d),
                "report_date": d.report_date.isoformat() if d.report_date else None,
                "transaction_date": d.transaction_date.isoformat() if d.transaction_date else None,
                "fingerprint": d.fingerprint,
            },
            "decision": asdict(decision) if decision else None,
        }
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def poll_once(self, price_returns: Optional[Dict[str, float]] = None) -> List[PelosiTailDecision]:
        disclosures = self.client.fetch_recent()
        seen = set(self._state.get("seen", []))

        # First start seeds the current API window.  Otherwise a brand-new install
        # could interpret weeks of previously disclosed history as fresh alerts.
        if not self._state.get("initialized", False) and self.seed_existing_on_first_run:
            for d in disclosures:
                seen.add(d.fingerprint)
                self._audit(d, None, "BASELINE_SEEDED")
            self._state["seen"] = sorted(seen)
            self._state["initialized"] = True
            self._state["baseline_seeded_at"] = _utc_now().isoformat()
            self._save_state()
            return []

        decisions: List[PelosiTailDecision] = []
        for d in sorted(disclosures, key=lambda x: (x.report_date or date.min, x.transaction_date or date.min, x.ticker)):
            if d.fingerprint in seen:
                continue
            now = _utc_now()
            drift = (price_returns or {}).get(d.ticker)
            decision = self.policy.decide(d, first_seen_at=now, trade_to_now_return=drift)
            decisions.append(decision)
            seen.add(d.fingerprint)
            self._audit(d, decision, "NEW_DISCLOSURE")

        self._state["seen"] = sorted(seen)[-5000:]
        self._state["initialized"] = True
        self._state["last_poll_at"] = _utc_now().isoformat()
        self._save_state()
        return decisions

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)


def run_poll_loop(
    poller: PelosiTailPoller,
    *,
    poll_seconds: float = 60.0,
    on_decision=None,
    stop_after_polls: int = 0,
) -> None:
    """Long-running poll loop with bounded backoff after source failures."""
    interval = max(15.0, float(poll_seconds))
    failures = 0
    polls = 0
    while True:
        polls += 1
        try:
            for decision in poller.poll_once():
                if on_decision:
                    on_decision(decision)
            failures = 0
            delay = interval
        except KeyboardInterrupt:
            return
        except Exception as exc:
            failures += 1
            delay = min(300.0, interval * (2 ** min(failures, 4)))
            print(f"Pelosi poll failed ({type(exc).__name__}: {exc}); retrying in {delay:.0f}s")
        if stop_after_polls and polls >= stop_after_polls:
            return
        time.sleep(delay)
