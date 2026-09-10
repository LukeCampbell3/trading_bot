"""V14.4 zero-wait warm-start utilities.

The warm-up requirement is preserved, but wall-clock waiting is removed by
reconstructing state from historical bars before live decisions are allowed.

Design goals:
- no look-ahead: every historical signal is generated from a prefix ending at
  that historical timestamp;
- checkpoint continuity: short restarts restore rolling bars and signal
  calibration, then backfill only the missing tail;
- fail-safe model identity: calibration checkpoints are ignored when model/
  scaler/feature identity changes;
- session-aware features for V14.3 options: ATR/trend may be seeded from prior
  completed bars, but VWAP/HOD/LOD are reset to the current regular session;
- overnight gaps never masquerade as 5m/15m continuation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque
import gzip
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd


WARM_START_VERSION = "14.4"
CHECKPOINT_SCHEMA = 1


@dataclass
class SymbolWarmStatus:
    symbol: str
    bars: int
    signals: int
    restored: bool
    replayed_signals: int
    ready: bool
    reason: str = ""


@dataclass
class WarmStartReport:
    version: str
    restored_checkpoint: bool
    model_fingerprint: str
    required_model_bars: int
    required_signal_history: int
    symbols: Dict[str, SymbolWarmStatus]

    @property
    def ready_count(self) -> int:
        return sum(1 for s in self.symbols.values() if s.ready)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ready_count"] = self.ready_count
        return out


class WarmStateStore:
    """Atomic gzip/json checkpoint store for rolling bars and calibration state."""

    def __init__(self, path: str = "state/v14_4_warm_state.json.gz", max_age_hours: float = 8.0):
        self.path = Path(path)
        self.max_age_hours = float(max_age_hours)

    @staticmethod
    def fingerprint(trader: Any) -> str:
        payload = {
            "version": getattr(trader, "VERSION", ""),
            "lookback": int(getattr(trader, "lookback", 0)),
            "seq_len": int(getattr(trader, "seq_len", 0)),
            "features": list(getattr(trader, "features", [])),
            "model_name": getattr(getattr(trader, "model", None), "name", None),
            "scaler_type": type(getattr(trader, "scaler", None)).__name__,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def _bar_payload(bar: Any) -> dict:
        return {
            "timestamp": pd.Timestamp(bar.ts).isoformat(),
            "open": float(bar.o),
            "high": float(bar.h),
            "low": float(bar.l),
            "close": float(bar.c),
            "volume": float(bar.v),
            "trade_count": float(bar.n),
            "vwap": float(bar.vwap),
        }

    def save(self, trader: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        symbols = {}
        for sym, state in getattr(trader, "sym_states", {}).items():
            bars = [self._bar_payload(b) for b in list(getattr(state.buf, "_dq", []))]
            symbols[sym] = {
                "bars": bars,
                "signal_history": [float(x) for x in list(state.signal_history)],
                "last_signal": None if state.last_signal is None else float(state.last_signal),
                "bars_seen": int(getattr(state, "bars_seen", 0)),
            }
        market_bars = [
            self._bar_payload(b) for b in list(getattr(getattr(trader, "buf_mkt", None), "_dq", []))
        ]
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "warm_start_version": WARM_START_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self.fingerprint(trader),
            "symbols": symbols,
            "market_bars": market_bars,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, self.path)

    def load(self, trader: Any) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("schema") != CHECKPOINT_SCHEMA:
                return None
            saved_at = pd.Timestamp(payload.get("saved_at"))
            if saved_at.tzinfo is None:
                saved_at = saved_at.tz_localize("UTC")
            age_hours = (pd.Timestamp.now(tz="UTC") - saved_at).total_seconds() / 3600.0
            if age_hours < 0 or age_hours > self.max_age_hours:
                return None
            if payload.get("fingerprint") != self.fingerprint(trader):
                return None
            return payload
        except Exception:
            return None

    @staticmethod
    def _restore_buffer(trader: Any, buf: Any, rows: List[dict]) -> int:
        # These are local state objects, not broker truth. Clearing before restore
        # avoids duplicate or out-of-order bars.
        if hasattr(buf, "_dq"):
            buf._dq.clear()
        if hasattr(buf, "_last_ts"):
            buf._last_ts = None
        count = 0
        for row in rows:
            try:
                bar = trader._row_to_bar(row)
                if buf.append_bar(bar):
                    count += 1
            except Exception:
                continue
        return count

    def restore(self, trader: Any) -> bool:
        payload = self.load(trader)
        if payload is None:
            return False
        for sym, item in payload.get("symbols", {}).items():
            state = getattr(trader, "sym_states", {}).get(sym)
            if state is None:
                continue
            self._restore_buffer(trader, state.buf, item.get("bars", []))
            state.signal_history.clear()
            for value in item.get("signal_history", [])[-state.signal_history.maxlen:]:
                if math.isfinite(float(value)):
                    state.signal_history.append(float(value))
            last = item.get("last_signal")
            state.last_signal = float(last) if last is not None and math.isfinite(float(last)) else None
            state.bars_seen = int(item.get("bars_seen", 0))
        mkt = getattr(trader, "buf_mkt", None)
        if mkt is not None:
            self._restore_buffer(trader, mkt, payload.get("market_bars", []))
        return True


class ZeroWaitWarmStarter:
    """Historical fast-forward of signal calibration with checkpoint continuity."""

    def __init__(
        self,
        checkpoint_path: str = "state/v14_4_warm_state.json.gz",
        required_signal_history: int = 30,
        checkpoint_max_age_hours: float = 8.0,
        delta_backfill_hours: float = 8.0,
    ):
        self.required_signal_history = int(required_signal_history)
        self.delta_backfill_hours = float(delta_backfill_hours)
        self.store = WarmStateStore(checkpoint_path, max_age_hours=checkpoint_max_age_hours)

    @staticmethod
    def required_model_bars(trader: Any) -> int:
        # Mirrors the current feature builder's hard information requirement.
        return int(trader.lookback + 25 + trader.seq_len)

    @staticmethod
    def _last_ts(buf: Any) -> Optional[pd.Timestamp]:
        value = getattr(buf, "last_timestamp", None)
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _backfill_one(self, trader: Any, symbol: str, buf: Any, full_limit: int) -> int:
        now = trader._get_server_utc_now()
        last = self._last_ts(buf)
        recent = False
        limit = int(full_limit)
        if last is not None:
            gap_hours = max(0.0, (now - last).total_seconds() / 3600.0)
            if gap_hours <= self.delta_backfill_hours:
                recent = True
                # Give the API enough overlap to survive missing minutes and dedupe.
                limit = min(int(full_limit), max(8, int(gap_hours * 60.0) + 12))
        df = trader._get_bars_rest(symbol, limit=limit, recent=recent)
        if df is None:
            return 0
        added = 0
        for _, row in df.iterrows():
            if buf.append_bar(trader._row_to_bar(row)):
                added += 1
        return added

    def restore_and_backfill(self, trader: Any) -> bool:
        restored = self.store.restore(trader)
        # Market tape first.
        self._backfill_one(trader, trader.market_symbol, trader.buf_mkt, trader._buffer_len)
        for sym, state in trader.sym_states.items():
            self._backfill_one(trader, sym, state.buf, trader._buffer_len)
        return restored

    @staticmethod
    def _temporary_buffer_like(buf: Any, bars: List[Any]) -> Any:
        temp = type(buf)(maxlen=max(getattr(buf, "maxlen", len(bars)), len(bars)))
        for bar in bars:
            temp.append_bar(bar)
        return temp

    def _historical_feature_windows(self, trader: Any, state: Any, needed: int):
        bars = list(getattr(state.buf, "_dq", []))
        minimum = self.required_model_bars(trader)
        if len(bars) < minimum or needed <= 0:
            return []
        # Take the most recent causal endpoints. Each temporary buffer ends at the
        # endpoint being labelled; later bars are never visible to that prediction.
        first_end = max(minimum, len(bars) - needed + 1)
        windows = []
        for end in range(first_end, len(bars) + 1):
            prefix = bars[:end]
            temp = self._temporary_buffer_like(state.buf, prefix)
            X = trader._compute_features_from_buffer(temp)
            if X is not None:
                windows.append((end, temp, X))
        return windows[-needed:]

    def _batch_predict(self, trader: Any, windows: List[Any]) -> List[float]:
        if not windows:
            return []
        if getattr(trader, "model", None) is None or getattr(trader, "scaler", None) is None:
            out = []
            for _, temp, _ in windows:
                value = float(trader._fallback_signal(temp))
                if math.isfinite(value):
                    out.append(value)
            return out

        X = np.stack([w[2] for w in windows]).astype(np.float32)
        n, seq, feat = X.shape
        X2 = X.reshape(-1, feat)
        X2s = trader.scaler.transform(X2)
        Xs = np.asarray(X2s, dtype=np.float32).reshape(n, seq, feat)
        pred = trader.model.predict(Xs, batch_size=min(256, max(1, n)), verbose=0)
        pred = np.asarray(pred).reshape(-1)
        return [float(v) for v in pred if math.isfinite(float(v))]

    def fast_forward_calibration(self, trader: Any) -> Dict[str, int]:
        replayed = {}
        for sym, state in trader.sym_states.items():
            # A valid checkpoint may already have a deep calibration history. Do
            # not duplicate it; otherwise reconstruct the exact minimum before live.
            missing = max(0, self.required_signal_history - len(state.signal_history))
            if missing <= 0:
                replayed[sym] = 0
                continue
            windows = self._historical_feature_windows(trader, state, missing)
            values = self._batch_predict(trader, windows)
            for value in values:
                state.signal_history.append(value)
            if values:
                state.last_signal = values[-1]
            replayed[sym] = len(values)
        return replayed

    def prepare(self, trader: Any) -> WarmStartReport:
        restored = self.restore_and_backfill(trader)
        replayed = self.fast_forward_calibration(trader)
        minimum = self.required_model_bars(trader)
        statuses = {}
        for sym, state in trader.sym_states.items():
            bars = len(state.buf)
            signals = len(state.signal_history)
            ready = bars >= minimum and signals >= self.required_signal_history
            reason = "ready" if ready else (
                f"need_bars:{minimum - bars}" if bars < minimum else
                f"need_signals:{self.required_signal_history - signals}"
            )
            statuses[sym] = SymbolWarmStatus(
                symbol=sym,
                bars=bars,
                signals=signals,
                restored=restored,
                replayed_signals=replayed.get(sym, 0),
                ready=ready,
                reason=reason,
            )
        # Persist the newly reconstructed state immediately so a crash/restart does
        # not pay the initialization cost again.
        try:
            self.store.save(trader)
        except Exception:
            pass
        return WarmStartReport(
            version=WARM_START_VERSION,
            restored_checkpoint=restored,
            model_fingerprint=self.store.fingerprint(trader),
            required_model_bars=minimum,
            required_signal_history=self.required_signal_history,
            symbols=statuses,
        )


class SessionWarmFeatureEngine:
    """Session-aware feature construction for the V14.3 options runner.

    Prior completed bars seed ATR/volatility and provide a conservative trend prior.
    Session-local VWAP/HOD/LOD and 5m/15m continuation never cross an overnight gap.
    """

    def __init__(self, timezone_name: str = "America/New_York", min_session_bars: int = 5):
        self.tz = timezone_name
        self.min_session_bars = int(min_session_bars)

    def compute(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            raise ValueError("no bars available")
        frame = df.copy().sort_index()
        if not isinstance(frame.index, pd.DatetimeIndex):
            if "timestamp" not in frame.columns:
                raise ValueError("bars need DatetimeIndex or timestamp column")
            frame = frame.set_index(pd.to_datetime(frame["timestamp"], utc=True))
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx_et = idx.tz_convert(self.tz)
        frame.index = idx

        close = frame["close"].astype(float).to_numpy()
        high = frame["high"].astype(float).to_numpy()
        low = frame["low"].astype(float).to_numpy()
        volume = frame["volume"].astype(float).to_numpy()
        if len(close) < 20:
            raise ValueError(f"need >=20 historical one-minute bars, got {len(close)}")

        current_day = idx_et[-1].date()
        hhmm = idx_et.hour * 60 + idx_et.minute
        session_mask = (
            np.asarray([d == current_day for d in idx_et.date]) &
            (hhmm >= 9 * 60 + 30) & (hhmm <= 16 * 60)
        )
        session = frame.iloc[np.where(session_mask)[0]]
        if session.empty:
            # Closed-market calls are informational only; never synthesize a session.
            raise ValueError("no current regular-session bars")

        sclose = session["close"].astype(float).to_numpy()
        shigh = session["high"].astype(float).to_numpy()
        slow = session["low"].astype(float).to_numpy()
        svol = session["volume"].astype(float).to_numpy()
        typical = (shigh + slow + sclose) / 3.0
        vwap = float(np.sum(typical * svol) / max(float(np.sum(svol)), 1.0))

        tr = np.zeros(len(close), dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr = max(float(np.mean(tr[-14:])), 1e-6)

        # Trend is session-first. Before 20 session bars exist, blend the short
        # session slope with the prior 20-bar slope rather than forcing a 20-minute wait.
        def slope(values: np.ndarray) -> float:
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values), dtype=float)
            return float(np.polyfit(x, values, 1)[0])

        prior_slope = slope(close[-20:])
        session_tail = sclose[-min(20, len(sclose)):]
        session_slope = slope(session_tail)
        weight = min(1.0, max(0.0, len(session_tail) / 20.0))
        trend_slope = weight * session_slope + (1.0 - weight) * prior_slope

        vol_tail = volume[-10:]
        vol_mean = float(np.mean(vol_tail)) if len(vol_tail) else 0.0
        current = float(sclose[-1])

        # Momentum must never cross the overnight boundary. Before enough session
        # bars exist, neutralize only that route feature instead of blocking the bot.
        p5 = float(sclose[-6]) if len(sclose) >= 6 else current
        p15 = float(sclose[-16]) if len(sclose) >= 16 else current

        return {
            "price": current,
            "vwap": vwap,
            "atr": atr,
            "high_of_day": float(np.max(shigh)),
            "low_of_day": float(np.min(slow)),
            "trend_slope": trend_slope,
            "volume_ratio": float(svol[-1] / vol_mean) if vol_mean > 0 else 1.0,
            "price_5m_ago": p5,
            "price_15m_ago": p15,
            "session_bars": int(len(sclose)),
            "trade_ready": bool(len(sclose) >= self.min_session_bars),
            "route_readiness": {
                "session_structure": bool(len(sclose) >= self.min_session_bars),
                "momentum_5m": bool(len(sclose) >= 6),
                "momentum_15m": bool(len(sclose) >= 16),
                "full_intraday_trend": bool(len(sclose) >= 20),
            },
        }
