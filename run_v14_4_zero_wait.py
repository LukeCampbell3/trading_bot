"""V14.4 zero-wait warm-start entrypoint for the core trader.

This preserves the trained model's 100-bar lookback/60-token sequence and the
30-observation signal calibration requirement. It removes wall-clock warm-up by
restoring a short-lived checkpoint or causally replaying historical prefixes
before the execution controller starts.

Examples:
    python run_v14_4_zero_wait.py --symbols COIN,TSLA
    python run_v14_4_zero_wait.py --symbols COIN --interval 30
"""

from __future__ import annotations

import argparse
import time

from alpaca_trader import AlpacaTrader
from strategy.warm_start_v14_4 import ZeroWaitWarmStarter


class V14_4_ZeroWaitTrader(AlpacaTrader):
    VERSION = "14.4-zero-wait"

    def __init__(
        self,
        *args,
        warm_state_path: str = "state/v14_4_warm_state.json.gz",
        warm_signal_count: int = 30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.warm_starter = ZeroWaitWarmStarter(
            checkpoint_path=warm_state_path,
            required_signal_history=warm_signal_count,
            checkpoint_max_age_hours=8.0,
            delta_backfill_hours=8.0,
        )
        self._warm_checkpoint_cycles = 0
        self._last_warm_report = None

    def bootstrap_buffers(self):
        """Replace live-time warm-up with restore -> delta backfill -> causal replay."""
        print(f"  V14.4 zero-wait warm start for {len(self.symbols)} symbols + SPY...")
        started = time.perf_counter()
        report = self.warm_starter.prepare(self)
        self._last_warm_report = report
        elapsed = time.perf_counter() - started
        not_ready = [s.symbol for s in report.symbols.values() if not s.ready]
        replayed = sum(s.replayed_signals for s in report.symbols.values())
        print(
            f"  Warm start complete in {elapsed:.2f}s | "
            f"ready={report.ready_count}/{len(report.symbols)} | "
            f"historical_signals={replayed} | "
            f"checkpoint_restored={report.restored_checkpoint}"
        )
        if not_ready:
            print(f"  Not ready (real data insufficient, fail-closed): {not_ready}")

    def generate_signals(self) -> list:
        # The parent now sees a populated signal_history on its first real-time call.
        signals = super().generate_signals()
        self._warm_checkpoint_cycles += 1
        # Persist frequently enough for restart continuity without writing each tick.
        if self._warm_checkpoint_cycles % 5 == 0:
            try:
                self.warm_starter.store.save(self)
            except Exception as exc:
                print(f"  Warm checkpoint warning: {exc}")
        return signals


def main() -> int:
    parser = argparse.ArgumentParser(description="V14.4 Zero-Wait Alpaca Trader")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated ticker list (default: full volatile universe)")
    parser.add_argument("--interval", type=int, default=45)
    parser.add_argument("--model", type=str, default="HFT/model/instinct_model.weights.h5")
    parser.add_argument("--scaler", type=str, default="HFT/model/scaler.pkl")
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--max-exposure", type=float, default=0.85)
    parser.add_argument("--warm-state", type=str, default="state/v14_4_warm_state.json.gz")
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    trader = V14_4_ZeroWaitTrader(
        symbols=symbols,
        model_path=args.model,
        scaler_path=args.scaler,
        warm_state_path=args.warm_state,
    )
    trader.max_concurrent_positions = args.max_positions
    trader.max_total_exposure = args.max_exposure
    trader.run(check_interval=args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
