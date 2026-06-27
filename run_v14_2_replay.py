"""
V14.2 Core Runner - Replay Harness Entry Point
===============================================
Status: V14_2_CORE_RUNNER_REPLACEMENT_SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN

Usage:
    python run_v14_2_replay.py [--data-file path/to/bars.csv] [--symbol SPY]

Runs the V14.2 replay harness on historical data to validate strategy performance
with timestamp-safe logic and option quote stress testing.

Required input CSV format (underlying bars):
    timestamp,open,high,low,close,volume,vwap

Optional option quote CSV format:
    timestamp,contract_symbol,bid,ask,volume,open_interest,iv,delta,strike,expiration,type

Output reports are written to: HFT/logs/v14_2/replay/
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from replay.real_option_quote_replay import (
    RealOptionQuoteReplay, ReplayBar, ReplayOptionQuote
)
from strategy.v14_2_config import V14_2_CORE_RUNNER_REPLACEMENT as CFG


def load_bars_from_csv(filepath: str) -> list:
    """Load underlying bars from CSV file."""
    bars = []
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        return bars

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                try:
                    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                except (ValueError, KeyError):
                    continue

            bars.append(ReplayBar(
                timestamp=ts,
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                vwap=float(row.get("vwap", row.get("close", 0))),
            ))

    print(f"[REPLAY] Loaded {len(bars)} bars from {filepath}")
    return bars


def load_option_quotes_from_csv(filepath: str) -> dict:
    """Load option quotes from CSV file."""
    quotes = {}
    path = Path(filepath)
    if not path.exists():
        print(f"[INFO] No option quotes file: {filepath}")
        return quotes

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue

            quote = ReplayOptionQuote(
                timestamp=ts,
                contract_symbol=row.get("contract_symbol", ""),
                bid=float(row.get("bid", 0)),
                ask=float(row.get("ask", 0)),
                mid=(float(row.get("bid", 0)) + float(row.get("ask", 0))) / 2,
                volume=int(row.get("volume", 0)),
                open_interest=int(row.get("open_interest", 0)),
                iv=float(row.get("iv", 0)),
                delta=float(row.get("delta", 0)),
                strike=float(row.get("strike", 0)),
                expiration=row.get("expiration", ""),
                option_type=row.get("type", "call"),
            )
            key = ts.strftime("%Y%m%d_%H%M")
            quotes.setdefault(key, []).append(quote)

    print(f"[REPLAY] Loaded option quotes for {len(quotes)} timestamps")
    return quotes


def generate_synthetic_bars(symbol: str = "SPY", count: int = 500) -> list:
    """Generate synthetic bars for testing when no real data is available."""
    import random
    random.seed(42)

    bars = []
    price = 450.0 if symbol == "SPY" else 150.0
    base_time = datetime(2025, 6, 1, 9, 30)

    from datetime import timedelta
    for i in range(count):
        change = random.gauss(0.02, 0.8)  # slight upward drift
        price += change
        high = price + abs(random.gauss(0, 0.3))
        low = price - abs(random.gauss(0, 0.3))
        bars.append(ReplayBar(
            timestamp=base_time + timedelta(minutes=i),
            open=price - change * 0.5,
            high=high,
            low=low,
            close=price,
            volume=50000 + random.randint(-10000, 10000),
            vwap=price - random.uniform(-0.2, 0.2),
        ))

    return bars


def main():
    parser = argparse.ArgumentParser(description="V14.2 Replay Harness")
    parser.add_argument("--data-file", default="", help="Path to underlying bars CSV")
    parser.add_argument("--option-file", default="", help="Path to option quotes CSV")
    parser.add_argument("--symbol", default="SPY", help="Symbol (default: SPY)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--bars", type=int, default=500, help="Synthetic bar count")
    args = parser.parse_args()

    print("=" * 72)
    print("V14.2 CORE RUNNER - REPLAY HARNESS")
    print("Status: SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN")
    print(f"Symbol: {args.symbol}")
    print("=" * 72)

    # Load data
    if args.data_file:
        bars = load_bars_from_csv(args.data_file)
    elif args.synthetic:
        print("[REPLAY] Using synthetic data (no real quote validation)")
        bars = generate_synthetic_bars(args.symbol, args.bars)
    else:
        print("[REPLAY] No data file specified. Use --data-file or --synthetic.")
        print("         Example: python run_v14_2_replay.py --data-file data/spy_1min.csv")
        print("         Or:      python run_v14_2_replay.py --synthetic --bars 1000")
        return

    option_quotes = {}
    if args.option_file:
        option_quotes = load_option_quotes_from_csv(args.option_file)

    # Run replay
    output_dir = "HFT/logs/v14_2/replay"
    harness = RealOptionQuoteReplay(
        underlying_bars=bars,
        option_quotes=option_quotes,
        symbol=args.symbol,
        output_dir=output_dir,
    )

    print(f"\n[REPLAY] Running replay on {len(bars)} bars...")
    session = harness.run()

    # Write reports
    harness.write_reports(session)

    # Print results
    print("\n" + "=" * 72)
    print("REPLAY RESULTS")
    print("=" * 72)
    print(f"  Period: {session.start_date} → {session.end_date}")
    print(f"  Total candidates scanned: {session.total_candidates}")
    print(f"  Watch tickets created: {session.total_watches}")
    print(f"  Confirmed: {session.total_confirmed}")
    print(f"  Filled: {session.total_filled}")
    print(f"  Missed fills: {session.total_missed}")
    print(f"  Closed: {session.total_closed}")
    print(f"")
    print(f"  Total P&L: ${session.total_pnl:.2f}")
    print(f"  Win rate: {session.win_rate:.1%}")
    print(f"  Profit factor: {session.profit_factor:.2f}")
    print(f"")
    print(f"  V14.2 total P&L: ${session.v14_2_total_pnl:.2f}")
    print(f"  V12.2 total P&L: ${session.v12_2_total_pnl:.2f}")

    # Acceptance gates
    print(f"\n{'─' * 72}")
    print("ACCEPTANCE GATES")
    print(f"{'─' * 72}")
    gates = [
        ("Win rate >= 58%", session.win_rate >= 0.58, f"{session.win_rate:.1%}"),
        ("Profit factor >= 2.0", session.profit_factor >= 2.0, f"{session.profit_factor:.2f}"),
        ("Total P&L positive", session.total_pnl > 0, f"${session.total_pnl:.2f}"),
        ("V14.2 beats V12.2", session.v14_2_total_pnl > session.v12_2_total_pnl,
         f"${session.v14_2_total_pnl:.2f} vs ${session.v12_2_total_pnl:.2f}"),
    ]
    all_passed = True
    for name, passed, value in gates:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name} → {value}")
        if not passed:
            all_passed = False

    print(f"\n  Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
    print(f"\n  Reports written to: {output_dir}/")
    print("=" * 72)

    if not all_passed:
        print("\n  NOTE: Failed gates with synthetic data is expected.")
        print("  Real quote validation requires timestamped option quote data.")
        print("  Status remains: SYNTHETIC_VALIDATED_NOT_REAL_QUOTE_PROVEN")


if __name__ == "__main__":
    main()
