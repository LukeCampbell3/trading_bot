"""Tests for V14.4 zero-wait warm-start behavior."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.warm_start_v14_4 import (
    SessionWarmFeatureEngine,
    WarmStateStore,
    ZeroWaitWarmStarter,
)


@dataclass
class FakeBar:
    ts: pd.Timestamp
    o: float
    h: float
    l: float
    c: float
    v: float
    n: float = 1.0
    vwap: float = 0.0


class FakeBuffer:
    def __init__(self, maxlen=80):
        self.maxlen = maxlen
        self._dq = deque(maxlen=maxlen)
        self._last_ts = None

    def __len__(self):
        return len(self._dq)

    @property
    def last_timestamp(self):
        return self._last_ts

    def append_bar(self, bar):
        if self._last_ts is not None and bar.ts <= self._last_ts:
            return False
        self._dq.append(bar)
        self._last_ts = bar.ts
        return True


class FakeScaler:
    def transform(self, x):
        return np.asarray(x)


class FakeModel:
    name = "fake"

    def predict(self, x, batch_size=None, verbose=0):
        # Prediction depends only on the last point visible in that historical prefix.
        return np.asarray(x)[:, -1, 0].reshape(-1, 1)


class FakeState:
    def __init__(self):
        self.buf = FakeBuffer(maxlen=80)
        self.signal_history = deque(maxlen=200)
        self.last_signal = None
        self.bars_seen = 0


class FakeTrader:
    VERSION = "test"
    lookback = 3
    seq_len = 2
    features = ["x"]
    _buffer_len = 80
    market_symbol = "SPY"

    def __init__(self):
        self.model = FakeModel()
        self.scaler = FakeScaler()
        self.sym_states = {"COIN": FakeState()}
        self.buf_mkt = FakeBuffer(maxlen=80)

    def _row_to_bar(self, r):
        return FakeBar(
            ts=pd.Timestamp(r["timestamp"]),
            o=float(r["open"]), h=float(r["high"]), l=float(r["low"]),
            c=float(r["close"]), v=float(r["volume"]),
            n=float(r.get("trade_count", 1.0)), vwap=float(r.get("vwap", r["close"])),
        )

    def _compute_features_from_buffer(self, buf):
        # Mirror the production information threshold: lookback + 25 + seq_len.
        if len(buf) < self.lookback + 25 + self.seq_len:
            return None
        closes = np.array([b.c for b in buf._dq], dtype=np.float32)
        return closes[-self.seq_len:].reshape(self.seq_len, 1)

    def _fallback_signal(self, buf):
        return float(buf._dq[-1].c)

    def _get_server_utc_now(self):
        return pd.Timestamp.now(tz="UTC")

    def _get_bars_rest(self, symbol, limit, recent=False):
        return None


def _seed_fake(trader, n=60):
    start = pd.Timestamp("2026-09-09T14:00:00Z")
    for i in range(n):
        px = float(i + 1)
        bar = FakeBar(start + pd.Timedelta(minutes=i), px, px, px, px, 1000.0, 1.0, px)
        trader.sym_states["COIN"].buf.append_bar(bar)
        trader.buf_mkt.append_bar(bar)


def test_historical_fast_forward_builds_30_causal_signals():
    trader = FakeTrader()
    _seed_fake(trader, 60)
    starter = ZeroWaitWarmStarter(required_signal_history=30)

    replayed = starter.fast_forward_calibration(trader)

    assert replayed["COIN"] == 30
    assert len(trader.sym_states["COIN"].signal_history) == 30
    # With 60 bars and 30 required outputs, causal endpoints are bars 31..60.
    # If future bars leaked into earlier predictions these would all equal 60.
    assert list(trader.sym_states["COIN"].signal_history) == [float(x) for x in range(31, 61)]
    assert trader.sym_states["COIN"].last_signal == 60.0


def test_checkpoint_restores_bars_and_calibration(tmp_path):
    source = FakeTrader()
    _seed_fake(source, 45)
    source.sym_states["COIN"].signal_history.extend([0.1, 0.2, 0.3])
    source.sym_states["COIN"].last_signal = 0.3
    store = WarmStateStore(str(tmp_path / "warm.json.gz"), max_age_hours=8)
    store.save(source)

    restored = FakeTrader()
    assert store.restore(restored) is True
    assert len(restored.sym_states["COIN"].buf) == 45
    assert len(restored.buf_mkt) == 45
    assert list(restored.sym_states["COIN"].signal_history) == [0.1, 0.2, 0.3]
    assert restored.sym_states["COIN"].last_signal == 0.3


def _frame(previous_prices, session_prices):
    rows = []
    idx = []
    prev_start = pd.Timestamp("2026-09-08 15:00", tz="America/New_York")
    for i, px in enumerate(previous_prices):
        idx.append((prev_start + pd.Timedelta(minutes=i)).tz_convert("UTC"))
        rows.append((px, px + 0.1, px - 0.1, px, 100.0))
    cur_start = pd.Timestamp("2026-09-09 09:30", tz="America/New_York")
    for i, px in enumerate(session_prices):
        idx.append((cur_start + pd.Timedelta(minutes=i)).tz_convert("UTC"))
        rows.append((px, px + 0.1, px - 0.1, px, 100.0))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["open", "high", "low", "close", "volume"])


def test_session_vwap_hod_lod_reset_and_prior_bars_seed_indicators():
    # Prior day is around 100, current day around 200. Session VWAP must not be ~150.
    df = _frame([100.0] * 30, [200.0, 201.0, 202.0, 203.0, 204.0, 205.0])
    out = SessionWarmFeatureEngine(min_session_bars=5).compute(df)

    assert out["trade_ready"] is True
    assert 201.0 < out["vwap"] < 204.0
    assert out["high_of_day"] > 205.0
    assert out["low_of_day"] < 200.0
    assert out["atr"] > 0
    assert out["route_readiness"]["momentum_5m"] is True


def test_overnight_gap_never_becomes_15m_momentum():
    df = _frame([100.0] * 30, [200.0, 201.0, 202.0, 203.0, 204.0, 205.0])
    out = SessionWarmFeatureEngine(min_session_bars=5).compute(df)

    assert out["route_readiness"]["momentum_15m"] is False
    assert out["price_15m_ago"] == out["price"]
    # 5m is allowed because six true current-session bars exist.
    assert out["price_5m_ago"] == 200.0


def test_insufficient_true_session_structure_remains_fail_closed():
    df = _frame([100.0] * 30, [200.0, 201.0, 202.0, 203.0])
    out = SessionWarmFeatureEngine(min_session_bars=5).compute(df)
    assert out["trade_ready"] is False
