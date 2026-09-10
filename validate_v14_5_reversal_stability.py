"""Deterministic V14.5 reversal-stability validation.

Measures how often a raw best-side selector would flip CALL/PUT compared with
V14.5's directional hysteresis.  This is a structural anti-whipsaw test, not a
profitability backtest and not evidence of live option performance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from strategy.options_reversal_guard import OptionsReversalGuard
from strategy.route_conditioning import RouteCandidate
from strategy.v14_5_stability_config import get_v14_5_config


@dataclass
class Point:
    call: float
    put: float
    price: float
    vwap: float
    atr: float = 1.0


def candidate(side: str, score: float) -> RouteCandidate:
    return RouteCandidate(
        route="VWAP_PULLBACK" if side == "CALL" else "PUT_REJECTION",
        side=side,
        score=score,
        ic_spread=0.04,
        ev_over_debit=0.20,
        option_liquidity=0.80,
        expected_move=2.0,
    )


def noisy_sequence() -> List[Point]:
    """Trend, noisy cross-signals, then a genuine persistent downside reversal."""
    seq = [
        Point(.66, .42, 101.0, 100.0),
        Point(.69, .44, 101.1, 100.0),
        Point(.72, .50, 101.2, 100.0),
        Point(.60, .64, 100.9, 100.0),  # one-bar PUT impulse
        Point(.68, .57, 101.0, 100.0),
        Point(.61, .65, 100.8, 100.0),  # another non-persistent impulse
        Point(.70, .54, 101.1, 100.0),
        Point(.59, .63, 100.7, 100.0),
        Point(.67, .55, 100.9, 100.0),
        # Genuine downside regime transition. Opposite edge and VWAP separation hold.
        Point(.50, .73, 99.7, 100.0),
        Point(.48, .75, 99.5, 100.0),
        Point(.46, .78, 99.3, 100.0),
        Point(.45, .80, 99.1, 100.0),
        Point(.52, .71, 99.4, 100.0),
    ]
    return seq


def raw_side(p: Point) -> str:
    return "CALL" if p.call >= p.put else "PUT"


def count_flips(sides: Iterable[str]) -> int:
    sides = [s for s in sides if s in ("CALL", "PUT")]
    return sum(1 for a, b in zip(sides, sides[1:]) if a != b)


def run(output: Path) -> dict:
    guard = OptionsReversalGuard(get_v14_5_config())
    rows = []
    raw = []
    guarded = []
    for i, p in enumerate(noisy_sequence(), 1):
        rs = raw_side(p)
        raw.append(rs)
        decision = guard.select_candidates(
            "COIN",
            [candidate("CALL", p.call), candidate("PUT", p.put)],
            price=p.price,
            vwap=p.vwap,
            atr=p.atr,
        )
        gs = guard.state("COIN").bias
        guarded.append(gs)
        rows.append({
            "bar": i,
            "call_score": p.call,
            "put_score": p.put,
            "raw_side": rs,
            "guarded_bias": gs,
            "guard_reason": decision.reason,
            "flip_streak": guard.state("COIN").flip_streak,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_direction_flips": count_flips(raw),
        "guarded_direction_flips": count_flips(guarded),
        "whipsaw_flips_avoided": count_flips(raw) - count_flips(guarded),
        "final_guarded_bias": guard.state("COIN").bias,
        "genuine_reversal_detected": guard.state("COIN").bias == "PUT",
        "output": str(output),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="HFT/logs/v14_5_options/reversal_stability.csv",
    )
    args = parser.parse_args()
    summary = run(Path(args.output))
    for key, value in summary.items():
        print(f"{key}={value}")
    if summary["guarded_direction_flips"] > 1:
        return 2
    if not summary["genuine_reversal_detected"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
