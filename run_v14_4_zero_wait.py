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
import hashlib
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Sort

from alpaca_trader import AlpacaTrader
from strategy.warm_start_v14_4 import ZeroWaitWarmStarter


def _artifact_identity(*paths) -> str:
    """Hash model/scaler bytes so stale calibration can never cross a model change."""
    h = hashlib.sha256()
    found = False
    for value in paths:
        if not value:
            continue
        path = Path(value)
        if not path.exists() or not path.is_file():
            continue
        found = True
        h.update(str(path).encode("utf-8"))
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()[:16] if found else "runtime-model"


class V14_4_ZeroWaitTrader(AlpacaTrader):
    VERSION = "14.4-zero-wait"

    def __init__(
        self,
        *args,
        warm_state_path: str = "state/v14_4_warm_state.json.gz",
        warm_signal_count: int = 30,
        **kwargs,
    ):
        model_path = kwargs.get("model_path")
        scaler_path = kwargs.get("scaler_path")
        artifact_id = _artifact_identity(model_path, scaler_path)
        super().__init__(*args, **kwargs)
        # WarmStateStore fingerprints VERSION. Include actual model/scaler bytes so
        # a replaced weights file invalidates prior calibration automatically.
        self.VERSION = f"14.4-zero-wait:{artifact_id}"
        self.warm_starter = ZeroWaitWarmStarter(
            checkpoint_path=warm_state_path,
            required_signal_history=warm_signal_count,
            checkpoint_max_age_hours=8.0,
            delta_backfill_hours=8.0,
        )
        self._warm_checkpoint_cycles = 0
        self._last_warm_report = None
        self._warm_ready_symbols = set()

    def _get_bars_rest(self, symbol: str, limit: int, recent: bool = False) -> Optional[pd.DataFrame]:
        """Always request newest bars first, then restore chronological order.

        The legacy bootstrap requests a seven-day range with a small limit but does
        not set sort direction. That can return an old page of the range instead of
        the newest context. Warm start must be anchored to the bar nearest startup.
        """
        now = self._get_server_utc_now().to_pydatetime()
        start = now - timedelta(hours=8) if recent else now - timedelta(days=7)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=now,
            limit=int(limit),
            feed=DataFeed.IEX,
            sort=Sort.DESC,
        )
        try:
            bars = self._call_data_api(self.data_client.get_stock_bars, req)
        except Exception as exc:
            if "not found" in str(exc).lower() or "no data" in str(exc).lower():
                return None
            raise
        df = bars.df
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level="symbol")
            except Exception:
                df = df.reset_index()
                df = df[df["symbol"] == symbol].set_index("timestamp")
        df = df.reset_index()
        expected = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
        if len(df.columns) == len(expected):
            df.columns = expected
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").tail(int(limit))

    def bootstrap_buffers(self):
        """Replace live-time warm-up with restore -> delta backfill -> causal replay."""
        print(f"  V14.4 zero-wait warm start for {len(self.symbols)} symbols + SPY...")
        started = time.perf_counter()
        report = self.warm_starter.prepare(self)
        self._last_warm_report = report
        self._warm_ready_symbols = {s.symbol for s in report.symbols.values() if s.ready}
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

    def _promote_newly_ready_symbols(self):
        """If an IPO/short-history symbol matures intraday, rebuild calibration causally."""
        minimum = self.warm_starter.required_model_bars(self)
        newly_ready = []
        for sym, state in self.sym_states.items():
            if sym in self._warm_ready_symbols or len(state.buf) < minimum:
                continue
            # Any fallback signals accumulated before full model readiness are not
            # admissible calibration evidence. Rebuild from real historical prefixes.
            state.signal_history.clear()
            state.last_signal = None
            newly_ready.append(sym)
        if newly_ready:
            self.warm_starter.fast_forward_calibration(self)
            for sym in newly_ready:
                state = self.sym_states[sym]
                if len(state.signal_history) >= self.warm_starter.required_signal_history:
                    self._warm_ready_symbols.add(sym)
                    print(f"  Warm-start promotion: {sym} became model/calibration ready")

    def generate_signals(self) -> list:
        self._promote_newly_ready_symbols()
        # The parent now sees a populated signal_history on its first real-time call.
        signals = super().generate_signals()
        minimum = self.warm_starter.required_model_bars(self)
        # Fail closed for any symbol that lacks either trained-model context or the
        # calibration distribution. This prevents the zero-wait layer from trading
        # on fabricated/default warm values.
        ready_now = {
            sym for sym, state in self.sym_states.items()
            if len(state.buf) >= minimum and
               len(state.signal_history) >= self.warm_starter.required_signal_history
        }
        self._warm_ready_symbols |= ready_now
        signals = [s for s in signals if getattr(s, "symbol", "") in self._warm_ready_symbols]

        self._warm_checkpoint_cycles += 1
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
