"""Run the Pelosi disclosure tail service.

Default mode is SHADOW_ONLY: detect new Nancy Pelosi disclosures, score the
remaining opportunity, and emit an auditable signal.  No broker order is sent.

The service is designed to stay alongside the stock trader as an alternative-data
signal producer.  It polls Quiver's live Congress feed and measures the public
filing lag explicitly so backtests cannot accidentally use the original trade date
as if it were observable in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pytz

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca_config import AlpacaConfig
from political_signals.pelosi_tail import (
    PelosiDisclosure,
    PelosiTailDecision,
    PelosiTailPolicy,
    PelosiTailPoller,
    QuiverCongressClient,
)


class AlpacaTradeDriftEstimator:
    """Estimate underlying drift from the disclosed transaction date to now.

    This is not an estimate of Pelosi's exact fill price.  Congressional reports do
    not publish exact execution prices.  We use the first available daily close on
    or after the disclosed transaction date and the newest one-minute close as a
    conservative residual-opportunity proxy.
    """

    def __init__(self):
        AlpacaConfig.validate()
        self.client = StockHistoricalDataClient(AlpacaConfig.API_KEY, AlpacaConfig.API_SECRET)
        self.eastern = pytz.timezone("America/New_York")

    def estimate(self, disclosure: PelosiDisclosure) -> Optional[float]:
        if not disclosure.transaction_date or not disclosure.ticker:
            return None
        try:
            start = self.eastern.localize(datetime.combine(disclosure.transaction_date, dt_time(4, 0)))
            end = datetime.now(self.eastern)
            if start >= end:
                return None
            daily_req = StockBarsRequest(
                symbol_or_symbols=disclosure.ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=10,
                feed=DataFeed.IEX,
            )
            daily = self.client.get_stock_bars(daily_req).df
            if daily.empty:
                return None
            if hasattr(daily.index, "nlevels") and daily.index.nlevels > 1:
                try:
                    daily = daily.xs(disclosure.ticker)
                except Exception:
                    pass
            anchor = float(daily["close"].iloc[0])
            if anchor <= 0:
                return None

            minute_req = StockBarsRequest(
                symbol_or_symbols=disclosure.ticker,
                timeframe=TimeFrame.Minute,
                start=max(start, end - timedelta(days=5)),
                end=end,
                limit=1,
                feed=DataFeed.IEX,
                sort="desc",
            )
            latest = self.client.get_stock_bars(minute_req).df
            if latest.empty:
                current = float(daily["close"].iloc[-1])
            else:
                if hasattr(latest.index, "nlevels") and latest.index.nlevels > 1:
                    try:
                        latest = latest.xs(disclosure.ticker)
                    except Exception:
                        pass
                current = float(latest["close"].iloc[0])
            if current <= 0:
                return None
            return current / anchor - 1.0
        except Exception:
            return None


class SignalSink:
    def __init__(self, output_dir: str = "HFT/logs/pelosi_tail"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.output_dir / "latest_signal.json"
        self.history_path = self.output_dir / "signals.jsonl"

    def emit(self, decision: PelosiTailDecision) -> None:
        payload = {
            "strategy": "PELOSI_DISCLOSURE_TAIL_V1",
            "mode": "SHADOW_ONLY",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            **decision.__dict__,
        }
        tmp = self.latest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.latest_path)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Track and tail Nancy Pelosi public stock disclosures")
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("PELOSI_POLL_SECONDS", "60")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--replay-existing", action="store_true", help="Process current Quiver window instead of baseline-seeding it")
    parser.add_argument("--no-alpaca-drift", action="store_true", help="Skip transaction-date price-drift estimate")
    parser.add_argument("--max-notional-pct", type=float, default=float(os.getenv("PELOSI_MAX_NOTIONAL_PCT", "0.08")))
    args = parser.parse_args()

    client = QuiverCongressClient()
    policy = PelosiTailPolicy(base_max_notional_pct=args.max_notional_pct)
    poller = PelosiTailPoller(
        client,
        policy,
        seed_existing_on_first_run=not args.replay_existing,
    )
    sink = SignalSink()

    drift_estimator = None
    if not args.no_alpaca_drift:
        try:
            drift_estimator = AlpacaTradeDriftEstimator()
        except Exception as exc:
            print(f"Alpaca drift estimator unavailable: {exc}; continuing without drift gating")

    polls = 0
    failures = 0
    interval = max(15.0, args.poll_seconds)
    print(
        "PELOSI_DISCLOSURE_TAIL_V1 | SHADOW_ONLY | "
        f"poll={interval:.0f}s | first-run-baseline={not args.replay_existing}"
    )

    while True:
        polls += 1
        try:
            # Fetch once so we can compute a per-ticker residual-opportunity estimate
            # before passing genuinely new disclosures through the policy.
            disclosures = client.fetch_recent()
            price_returns: Dict[str, float] = {}
            if drift_estimator:
                for d in disclosures:
                    if d.ticker and d.ticker not in price_returns:
                        drift = drift_estimator.estimate(d)
                        if drift is not None:
                            price_returns[d.ticker] = drift

            # Avoid a second HTTP request by temporarily serving this fetched window.
            original = client.fetch_recent
            client.fetch_recent = lambda: disclosures
            try:
                decisions = poller.poll_once(price_returns=price_returns)
            finally:
                client.fetch_recent = original

            for decision in decisions:
                sink.emit(decision)
            failures = 0
            if args.once:
                return 0
            time.sleep(interval)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            failures += 1
            delay = min(300.0, interval * (2 ** min(failures, 4)))
            print(f"Poll failed: {type(exc).__name__}: {exc}; retrying in {delay:.0f}s")
            if args.once:
                return 1
            time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())
