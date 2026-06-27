"""
alpaca_trader.py  --  v14.2

Multi-Ticker Alpaca Paper Trading Bot (Transformer/sequence-ready)
==================================================================

Version 14.2 Changes
--------------------
- Multi-symbol universe: trades across 40+ volatile tickers simultaneously
- Per-symbol rolling buffers, signal history, and position state
- Adjusted liquidity bounds for IEX free-tier (relaxed min_volume_gate, adaptive
  spread proxy per-ticker, vol-scaled cost estimation)
- Round-robin polling with rate-aware batching
- Portfolio-level risk: max total exposure, per-symbol notional cap, correlated
  position limit
- Daily universe rotation: ranks symbols by recent realized vol, prunes dead tickers

Core improvements (all 10 from prior versions, preserved)
---------------------------------------------------------
1) Streaming-first + rolling in-memory bar buffer (fallback to REST polling)
2) Cost-aware execution gating (spread/impact proxies)
3) Bracket orders (stop + take profit) with broker-enforced risk
4) True volatility-targeted sizing (risk-per-trade sizing using stop distance)
5) Regime filter (vol/chop/liquidity + time-of-day gating)
6) Independent confirmations (market tape via SPY; range expansion/compression)
7) Incremental/rolling computation (rolling buffers, no big df recompute)
8) Trading hygiene (daily max loss kill-switch, max trades/day, stale-data skip)
9) Output calibration (adaptive threshold via signal percentiles + mean/std)
10) Online evaluation loop (CSV logging, rolling stats, adaptive tightening)

Notes
-----
- Model input shape: (1, SEQ_LEN, n_features).
- Uses IEX feed for bars (free). If you have SIP, change feed parameter.
"""

from __future__ import annotations

import os
import time
import math
import csv
import re
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import zipfile
import tempfile
import h5py

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Bracket orders (optional; fall back to non-bracket orders if unavailable)
try:
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
    _BRACKET_OK = True
except Exception:
    OrderClass = None
    TakeProfitRequest = None
    StopLossRequest = None
    _BRACKET_OK = False

from alpaca_config import AlpacaConfig


# ============================================================================
# VOLATILE STOCK UNIVERSE (40+ tickers)
# High-beta, liquid names suited for intraday momentum/mean-reversion.
# Curated for IEX free-tier viability (sufficient IEX volume during market hours).
# ============================================================================

VOLATILE_UNIVERSE = [
    # Mega-cap tech (high intraday range, massive liquidity)
    "NVDA", "TSLA", "AMD", "META", "AMZN", "AAPL", "GOOGL", "MSFT", "NFLX", "AVGO",
    # Semis & high-beta tech
    "SMCI", "MRVL", "MU", "QCOM", "ARM", "CRWD", "PLTR", "SNOW", "NET", "DDOG",
    # Volatile growth / momentum
    "COIN", "MSTR", "SHOP", "XYZ", "ROKU", "RBLX", "HOOD", "SOFI", "UPST", "AFRM",
    # Energy / commodities (vol spikes)
    "XOM", "OXY", "FSLR", "ENPH",
    # Biotech / pharma (gap movers)
    "MRNA", "BNTX",
    # Leveraged ETFs (inherent volatility, liquid on IEX)
    "TQQQ", "SOXL", "SQQQ", "SPXL",
    # High-vol mid-caps
    "RIVN", "LCID", "NIO", "IONQ", "RGTI",
]

# De-duplicate and sort for deterministic ordering
VOLATILE_UNIVERSE = sorted(set(VOLATILE_UNIVERSE))


# ============================================================================
# Custom Keras layers (must match training)
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        key_dim = max(1, self.d_model // max(1, self.num_heads))
        self.attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.d_model),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(self.dropout)
        self.drop2 = layers.Dropout(self.dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.ln1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.ln2(x + self.drop2(ffn_out, training=training))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim, "dropout": self.dropout})
        return cfg


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.pos_emb = layers.Embedding(input_dim=self.seq_len, output_dim=self.d_model)

    def call(self, x):
        T = tf.shape(x)[1]
        positions = tf.range(start=0, limit=T, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


# ============================================================================
# Rolling buffers & helpers
# ============================================================================

@dataclass
class Bar:
    ts: pd.Timestamp
    o: float
    h: float
    l: float
    c: float
    v: float
    n: float
    vwap: float


class RollingBarBuffer:
    """In-memory rolling buffer for 1m bars."""
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._dq: Deque[Bar] = deque(maxlen=maxlen)
        self._last_ts: Optional[pd.Timestamp] = None

    def __len__(self) -> int:
        return len(self._dq)

    @property
    def last_timestamp(self) -> Optional[pd.Timestamp]:
        return self._last_ts

    def append_bar(self, bar: Bar) -> bool:
        if self._last_ts is not None and bar.ts <= self._last_ts:
            return False
        self._dq.append(bar)
        self._last_ts = bar.ts
        return True

    def to_arrays(self) -> Dict[str, np.ndarray]:
        if not self._dq:
            return {}
        return {
            "ts": np.array([b.ts.value for b in self._dq], dtype=np.int64),
            "open": np.array([b.o for b in self._dq], dtype=np.float64),
            "high": np.array([b.h for b in self._dq], dtype=np.float64),
            "low": np.array([b.l for b in self._dq], dtype=np.float64),
            "close": np.array([b.c for b in self._dq], dtype=np.float64),
            "volume": np.array([b.v for b in self._dq], dtype=np.float64),
            "trade_count": np.array([b.n for b in self._dq], dtype=np.float64),
            "vwap": np.array([b.vwap for b in self._dq], dtype=np.float64),
        }

    def last_close(self) -> Optional[float]:
        return None if not self._dq else float(self._dq[-1].c)


def _safe_float(x, default=0.0) -> float:
    try:
        val = float(x)
        return val if math.isfinite(val) else default
    except Exception:
        return default


@dataclass
class SymbolState:
    """Per-symbol mutable state for multi-ticker trading."""
    symbol: str
    buf: RollingBarBuffer = field(default=None)
    signal_history: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    last_signal: Optional[float] = None
    position_bars_held: int = 0
    best_pnl_pct: Optional[float] = None
    cooldown_bars_remaining: int = 0
    exit_weak_streak: int = 0
    bars_seen: int = 0
    last_processed_ts: Optional[pd.Timestamp] = None
    exit_order_pending: bool = False
    exit_order_id: str = ""
    exit_pending_notice_count: int = 0
    recent_trade_pnls: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    recent_wins: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    # Adaptive params per symbol
    signal_percentile: float = 0.65
    risk_per_trade: float = 0.0020

    def __post_init__(self):
        if self.buf is None:
            self.buf = RollingBarBuffer(maxlen=220)


# ============================================================================
# Model builder (must match training.py architecture)
# ============================================================================

def build_model(seq_len: int, n_features: int,
                d_model: int = 64, n_heads: int = 4, ff_dim: int = 128,
                n_blocks: int = 3, dropout: float = 0.15):
    inp = layers.Input(shape=(seq_len, n_features), name="x")
    x = layers.Dense(d_model, name="proj")(inp)
    x = PositionalEmbedding(seq_len, d_model, name="pos")(x)
    x = layers.Dropout(dropout, name="drop_in")(x)
    for i in range(n_blocks):
        x = TransformerBlock(d_model=d_model, num_heads=n_heads, ff_dim=ff_dim,
                             dropout=dropout, name=f"tb{i}")(x)
    last = layers.Lambda(lambda t: t[:, -1, :], name="last_tok")(x)
    avg = layers.GlobalAveragePooling1D(name="avg_pool")(x)
    x = layers.Concatenate(name="pool_cat")([last, avg])
    x = layers.Dense(128, activation="gelu", name="head_128")(x)
    x = layers.Dropout(dropout, name="drop_128")(x)
    x = layers.Dense(64, activation="gelu", name="head_64")(x)
    x = layers.Dropout(dropout, name="drop_64")(x)
    out = layers.Dense(1, activation="linear", name="y")(x)
    return keras.Model(inp, out, name="instinct_transformer")


# ============================================================================
# AlpacaTrader  --  v14.2 Multi-Ticker
# ============================================================================

class AlpacaTrader:
    """Multi-ticker paper trading bot using Alpaca free API (v14.2)."""

    VERSION = "14.2"

    def __init__(self, symbols: List[str] = None, model_path=None, scaler_path=None):
        AlpacaConfig.validate()

        # Universe
        self.symbols = symbols or VOLATILE_UNIVERSE
        self.market_symbol = "SPY"
        self.eastern = pytz.timezone('US/Eastern')

        # Session
        self.enable_extended_hours = True
        self.regular_open_hm = (9, 30)
        self.regular_close_hm = (16, 0)
        self.extended_open_hm = (4, 0)
        self.extended_close_hm = (20, 0)
        self.enforce_freshness_when_polling = False

        # Clients
        self._init_clients()
        self.max_data_age_regular_sec = 180
        self.max_data_age_extended_sec = 480
        self._server_time_ttl_seconds = 20
        self._last_server_time_fetch = 0.0
        self._last_server_timestamp: Optional[pd.Timestamp] = None
        self._last_local_timestamp_at_server_fetch: Optional[pd.Timestamp] = None

        # Model / features
        self.lookback = 100
        self.seq_len = 60
        self.features = ["z", "dz", "avg_dz", "ma_slope", "vol_surge", "dev_score", "trend"]
        self._buffer_len = self.lookback + self.seq_len + 60

        self.model = None
        self.scaler = None
        self._load_model_and_scaler(model_path, scaler_path)

        # Per-symbol state
        self.sym_states: Dict[str, SymbolState] = {}
        for sym in self.symbols:
            self.sym_states[sym] = SymbolState(symbol=sym, buf=RollingBarBuffer(maxlen=self._buffer_len))
        # Market tape buffer (SPY)
        self.buf_mkt = RollingBarBuffer(maxlen=self._buffer_len)

        # ── Risk & sizing (v14.2 adjusted) ────────────────────────────────
        self.vol_window = 60
        self.base_risk_per_trade = 0.0035       # 0.35% equity per trade
        self.max_notional_fraction = 0.18       # per-symbol cap
        self.max_total_exposure = 0.85          # max 85% of equity deployed across all
        self.max_concurrent_positions = 8       # correlation risk cap
        self.min_qty = 1
        self.max_qty = 5000

        # Vol-based stops/targets
        self.stop_k = 2.2
        self.take_k = 3.2
        self.trail_activation_k = 1.5
        self.trail_giveback_k = 0.9

        # ── Entry calibration & cost (v14.2 liquidity-relaxed) ────────────
        self.base_entry_threshold = 0.00008
        self.base_exit_threshold = 0.00010
        self.entry_zscore_sigma = 0.35
        self.base_signal_percentile = 0.65
        self.min_edge_over_cost = 0.00003
        self.entry_sig_z_main = 1.7
        self.entry_sig_z_persist = 1.4
        self.entry_sig_z_override = 2.0
        self.exit_sig_z_soft = 0.2
        self.exit_sig_z_hard = 0.0
        self.exit_confirm_bars = 2
        self.tape_long_min_5m = -0.0010
        self.feature_checksum_every_bars = 50

        # ── Liquidity bounds (v14.2 -- relaxed for IEX free tier) ─────────
        # IEX shows ~5-15% of total market volume. For volatile names, even
        # 500 shares/min on IEX is tradeable on Alpaca (orders route to best ex).
        # The key insight: IEX volume is used for SIGNAL quality, not order routing.
        # Alpaca routes to NBBO regardless of data feed.
        self.vol_hi_gate = 0.0050              # raised: allow more volatile names
        self.min_volume_gate = 200             # relaxed from 5000 -- IEX shows fraction
        self.range_chop_gate = 0.0060          # raised: volatile names have wider bars
        self.min_dollar_volume_gate = 50_000   # NEW: $vol floor (price * vol > this)

        # ── Regime & hygiene ──────────────────────────────────────────────
        self.no_trade_open_minutes = 0
        self.no_trade_close_minutes = 2
        self.max_trades_per_day = 60           # raised for multi-ticker
        self.daily_max_drawdown = 0.035        # 3.5% portfolio drawdown kill-switch
        self.pause_minutes_on_kill = 30
        self.reentry_cooldown_bars = 1
        self.early_no_follow_bars = 4
        self.no_follow_progress_k = 0.35
        self.time_stop_bars = 8
        self.edge_gone_min_hold_bars = 3
        self.signal_reversal_delta = 0.000015

        # ── Global state ──────────────────────────────────────────────────
        self._today = None
        self._trades_today = 0
        self._day_start_equity = None
        self._paused_until: Optional[datetime] = None

        # Rate pacing
        self._rate_window_seconds = 60
        self._trading_limit_per_min = 120
        self._data_limit_per_min = 180         # raised for multi-ticker polling
        self._trading_call_timestamps: Deque[float] = deque()
        self._data_call_timestamps: Deque[float] = deque()

        # Account cache
        self._cache_ttl_seconds = 8
        self._last_account_fetch = 0.0
        self._account_cache = None

        # Logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log_path = self.log_dir / "trades_multi.csv"
        self.state_log_path = self.log_dir / "state_multi.csv"
        self._init_logs()

    # ──────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────
    def _load_model_and_scaler(self, model_path, scaler_path):
        mp = Path(model_path) if model_path else None
        sp = Path(scaler_path) if scaler_path else None

        keras_path = None
        weights_path = None
        if mp:
            if mp.suffix == ".keras":
                keras_path = mp
                cand = mp.with_suffix(".weights.h5")
                if cand.exists():
                    weights_path = cand
            elif mp.suffix in [".h5", ".weights"]:
                weights_path = mp
                cand = mp.with_suffix(".keras")
                if cand.exists():
                    keras_path = cand

        # Fallbacks
        for base in [Path("HFT/model"), Path("model")]:
            if keras_path is None and (base / "instinct_model.keras").exists():
                keras_path = base / "instinct_model.keras"
            if weights_path is None and (base / "instinct_model.weights.h5").exists():
                weights_path = base / "instinct_model.weights.h5"
            if sp is None and (base / "scaler.pkl").exists():
                sp = base / "scaler.pkl"

        if sp and sp.exists():
            self.scaler = joblib.load(sp)
            print(f"  Scaler loaded from {sp}")
        else:
            print("  Scaler not found.")

        if keras_path and keras_path.exists():
            try:
                self.model = keras.models.load_model(
                    str(keras_path),
                    custom_objects={"TransformerBlock": TransformerBlock,
                                    "PositionalEmbedding": PositionalEmbedding},
                    compile=False, safe_mode=False)
                print(f"  Model loaded from .keras archive.")
            except Exception as e:
                print(f"  .keras load failed ({e}). Trying weight extraction...")
                try:
                    self.model = self._load_model_via_weights_archive(str(keras_path))
                    print("  Model rebuilt + archive weights loaded.")
                except Exception as e2:
                    print(f"  Archive extraction failed ({e2}).")
                    self.model = None

        if self.model is None and weights_path and weights_path.exists():
            print(f"  Loading model from weights: {weights_path}")
            self.model = build_model(seq_len=self.seq_len, n_features=len(self.features))
            dummy = tf.zeros((1, self.seq_len, len(self.features)), dtype=tf.float32)
            _ = self.model(dummy, training=False)
            try:
                self.model.load_weights(str(weights_path))
                print("  Model rebuilt + weights loaded.")
            except Exception as e:
                self._load_weights_h5_compat(self.model, str(weights_path))
                print("  Model rebuilt + HDF5 compat weights loaded.")

        if self.model is None:
            print("  WARNING: No model loaded. Will use fallback signal.")

    def _load_model_via_weights_archive(self, model_path: str):
        mp = Path(model_path)
        if not mp.exists():
            return None
        m = build_model(seq_len=self.seq_len, n_features=len(self.features))
        cache_dir = Path("model/_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_file = cache_dir / f"{mp.stem}.extracted.weights.h5"
        with zipfile.ZipFile(mp, "r") as z:
            if "model.weights.h5" not in z.namelist():
                raise ValueError("model.weights.h5 not found in .keras archive")
            with z.open("model.weights.h5") as src, open(weights_file, "wb") as dst:
                dst.write(src.read())
        dummy = tf.zeros((1, self.seq_len, len(self.features)), dtype=tf.float32)
        _ = m(dummy, training=False)
        try:
            m.load_weights(str(weights_file), by_name=False, skip_mismatch=False)
        except Exception:
            self._load_weights_h5_compat(m, str(weights_file))
        return m

    @staticmethod
    def _h5_group_vars(group) -> list:
        out = []
        if group is None or "vars" not in group:
            return out
        vg = group["vars"]
        for k in sorted(vg.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            out.append(np.array(vg[k]))
        return out

    def _load_weights_h5_compat(self, model, weights_path: str):
        with h5py.File(weights_path, "r") as f:
            root = f.get("_layer_checkpoint_dependencies")
            if root is None:
                raise ValueError("Unsupported HDF5 format")
            model.get_layer("proj").set_weights(self._h5_group_vars(root["dense"]))
            model.get_layer("pos").pos_emb.set_weights(
                self._h5_group_vars(root["positional_embedding"]["pos_emb"]))
            model.get_layer("head_128").set_weights(self._h5_group_vars(root["dense_1"]))
            model.get_layer("head_64").set_weights(self._h5_group_vars(root["dense_2"]))
            model.get_layer("y").set_weights(self._h5_group_vars(root["dense_3"]))
            block_names = ["transformer_block", "transformer_block_1", "transformer_block_2"]
            for i, fn in enumerate(block_names):
                if fn not in root:
                    raise ValueError(f"Missing block: {fn}")
                bg = root[fn]
                blk = model.get_layer(f"tb{i}")
                ag = bg["attn"]
                attn_w = []
                for sub in ["_query_dense", "_key_dense", "_value_dense", "_output_dense"]:
                    attn_w.extend(self._h5_group_vars(ag[sub]))
                blk.attn.set_weights(attn_w)
                ffn_deps = bg["ffn"]["_layer_checkpoint_dependencies"]
                ffn_w = []
                for sub in ["dense", "dense_1"]:
                    ffn_w.extend(self._h5_group_vars(ffn_deps[sub]))
                blk.ffn.set_weights(ffn_w)
                blk.ln1.set_weights(self._h5_group_vars(bg["ln1"]))
                blk.ln2.set_weights(self._h5_group_vars(bg["ln2"]))

    # ──────────────────────────────────────────────────────────────────────
    # Alpaca API clients + pacing
    # ──────────────────────────────────────────────────────────────────────
    def _init_clients(self):
        self.trading_client = TradingClient(
            AlpacaConfig.API_KEY, AlpacaConfig.API_SECRET,
            paper=AlpacaConfig.PAPER, url_override=AlpacaConfig.BASE_URL)
        self.data_client = StockHistoricalDataClient(
            AlpacaConfig.API_KEY, AlpacaConfig.API_SECRET,
            url_override=AlpacaConfig.DATA_URL)

    def _call_api(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if not any(t in msg for t in ("connection", "timeout", "tempor", "reset", "502", "503", "504")):
                raise
            print(f"  API retry ({e})...")
            self._init_clients()
            return fn(*args, **kwargs)

    def _pace_api(self, bucket_name):
        now = time.time()
        if bucket_name == "trading":
            bucket, limit = self._trading_call_timestamps, self._trading_limit_per_min
        else:
            bucket, limit = self._data_call_timestamps, self._data_limit_per_min
        cutoff = now - self._rate_window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            sleep_for = (bucket[0] + self._rate_window_seconds) - now + 0.25
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.time()
            while bucket and bucket[0] < (now - self._rate_window_seconds):
                bucket.popleft()
        bucket.append(time.time())

    def _sleep_on_rate_limit(self, error, default_seconds=60):
        msg = str(error).lower()
        if "429" not in msg and "rate limit" not in msg:
            return False
        print(f"  429 rate limit. Sleeping {default_seconds}s...")
        time.sleep(default_seconds)
        return True

    def _call_trading_api(self, fn, *args, **kwargs):
        self._pace_api("trading")
        try:
            return self._call_api(fn, *args, **kwargs)
        except Exception as e:
            if self._sleep_on_rate_limit(e):
                self._pace_api("trading")
                return self._call_api(fn, *args, **kwargs)
            raise

    def _call_data_api(self, fn, *args, **kwargs):
        self._pace_api("data")
        try:
            return self._call_api(fn, *args, **kwargs)
        except Exception as e:
            if self._sleep_on_rate_limit(e, default_seconds=30):
                self._pace_api("data")
                return self._call_api(fn, *args, **kwargs)
            raise

    def _get_server_utc_now(self, force: bool = False) -> pd.Timestamp:
        local_now = pd.Timestamp.now(tz="UTC")
        if (not force) and self._last_server_timestamp is not None and \
           (time.time() - self._last_server_time_fetch) < self._server_time_ttl_seconds:
            elapsed = local_now - self._last_local_timestamp_at_server_fetch
            return self._last_server_timestamp + elapsed
        try:
            clock = self._call_trading_api(self.trading_client.get_clock)
            raw_ts = getattr(clock, "timestamp", None) or getattr(clock, "current_time", None)
            if raw_ts is not None:
                server_now = pd.to_datetime(raw_ts, utc=True)
                if pd.notna(server_now):
                    self._last_server_timestamp = server_now
                    self._last_local_timestamp_at_server_fetch = local_now
                    self._last_server_time_fetch = time.time()
                    return server_now
        except Exception:
            pass
        return local_now

    # ──────────────────────────────────────────────────────────────────────
    # Account / positions
    # ──────────────────────────────────────────────────────────────────────
    def get_account_info(self, force=False):
        now = time.time()
        if not force and self._account_cache and (now - self._last_account_fetch) < self._cache_ttl_seconds:
            return self._account_cache
        account = self._call_trading_api(self.trading_client.get_account)
        info = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }
        self._account_cache = info
        self._last_account_fetch = now
        return info

    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            p = self._call_trading_api(self.trading_client.get_open_position, symbol)
            return {
                "qty": int(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
        except Exception:
            return None

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all open positions at once (1 API call)."""
        try:
            positions = self._call_trading_api(self.trading_client.get_all_positions)
            result = {}
            for p in positions:
                result[p.symbol] = {
                    "qty": int(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
            return result
        except Exception:
            return {}

    # ──────────────────────────────────────────────────────────────────────
    # Data ingestion (multi-symbol REST polling)
    # ──────────────────────────────────────────────────────────────────────
    def _row_to_bar(self, r) -> Bar:
        ts = pd.to_datetime(r["timestamp"], utc=True)
        return Bar(ts=ts, o=_safe_float(r["open"]), h=_safe_float(r["high"]),
                   l=_safe_float(r["low"]), c=_safe_float(r["close"]),
                   v=_safe_float(r["volume"]), n=_safe_float(r.get("trade_count", 0.0)),
                   vwap=_safe_float(r.get("vwap", r.get("close", 0.0))))

    def _get_bars_rest(self, symbol: str, limit: int, recent: bool = False) -> Optional[pd.DataFrame]:
        now = self._get_server_utc_now().to_pydatetime()
        start = now - timedelta(hours=8) if recent else now - timedelta(days=7)
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
            start=start, end=now, limit=limit, feed=DataFeed.IEX)
        try:
            bars = self._call_data_api(self.data_client.get_stock_bars, req)
        except Exception as e:
            if "not found" in str(e).lower() or "no data" in str(e).lower():
                return None
            raise
        df = bars.df
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df = df.reset_index()
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").tail(limit)

    def _get_bars_multi(self, symbols: List[str], limit: int, recent: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols. Uses per-symbol requests for reliability
        (multi-symbol requests share the limit quota across all symbols)."""
        result = {}
        for sym in symbols:
            df = self._get_bars_rest(sym, limit, recent=recent)
            if df is not None and len(df) > 0:
                result[sym] = df
        return result

    def bootstrap_buffers(self):
        """Pull initial bars for all symbols + SPY. Uses batched multi-symbol requests."""
        print(f"  Bootstrapping {len(self.symbols)} symbols + SPY...")
        # SPY (market tape)
        mkt_df = self._get_bars_rest(self.market_symbol, self._buffer_len)
        if mkt_df is not None:
            for _, r in mkt_df.iterrows():
                self.buf_mkt.append_bar(self._row_to_bar(r))
        print(f"    SPY buffer: {len(self.buf_mkt)} bars")

        # Batch symbols in groups of 10 (Alpaca multi-symbol limit is generous but
        # we keep batches modest to avoid timeouts)
        batch_size = 10
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            dfs = self._get_bars_multi(batch, self._buffer_len, recent=False)
            for sym, df in dfs.items():
                if sym in self.sym_states:
                    for _, r in df.iterrows():
                        self.sym_states[sym].buf.append_bar(self._row_to_bar(r))
            loaded = [s for s in batch if s in dfs]
            empty = [s for s in batch if s not in dfs]
            if empty:
                print(f"    No data: {empty}")
            time.sleep(0.3)  # gentle pacing

        active = [s for s, st in self.sym_states.items() if len(st.buf) >= self.lookback]
        print(f"  Bootstrap complete. {len(active)}/{len(self.symbols)} symbols ready.")

    def poll_latest(self):
        """Poll latest bars for all active symbols + SPY."""
        # SPY
        mkt_df = self._get_bars_rest(self.market_symbol, limit=4, recent=True)
        if mkt_df is not None:
            for _, r in mkt_df.iterrows():
                self.buf_mkt.append_bar(self._row_to_bar(r))

        # Batch poll symbols
        batch_size = 10
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            dfs = self._get_bars_multi(batch, limit=4, recent=True)
            for sym, df in dfs.items():
                if sym in self.sym_states:
                    for _, r in df.iterrows():
                        self.sym_states[sym].buf.append_bar(self._row_to_bar(r))
            time.sleep(0.2)

    # ──────────────────────────────────────────────────────────────────────
    # Feature computation on rolling buffers
    # ──────────────────────────────────────────────────────────────────────
    def _compute_features_from_buffer(self, buf: RollingBarBuffer) -> Optional[np.ndarray]:
        """Returns shape (seq_len, n_features) or None."""
        arr = buf.to_arrays()
        if not arr:
            return None
        close = arr["close"]
        volume = arr["volume"]
        need_min = self.lookback + 25 + self.seq_len
        if len(close) < need_min:
            return None

        tail = close[-(self.lookback + self.seq_len + 40):]
        vtail = volume[-(self.lookback + self.seq_len + 40):]
        if len(tail) < (self.lookback + self.seq_len + 20):
            return None

        pxv = tail * vtail
        vw_sum = pd.Series(pxv).rolling(self.lookback).sum().to_numpy()
        v_sum = pd.Series(vtail).rolling(self.lookback).sum().to_numpy()
        vwap = vw_sum / np.where(v_sum == 0, np.nan, v_sum)
        std = pd.Series(tail).rolling(self.lookback).std().to_numpy()

        z = (tail - vwap) / std
        prev_z = np.roll(z, 1); prev_z[0] = np.nan
        dz = z - prev_z
        avg_dz = pd.Series(dz).rolling(5).mean().to_numpy()

        ma_recent = pd.Series(tail).rolling(20).mean().to_numpy()
        ma_prev = pd.Series(np.roll(tail, 10)).rolling(20).mean().to_numpy()
        ma_prev[:10] = np.nan
        ma_slope = ma_recent - ma_prev

        vol_mean = pd.Series(vtail).rolling(10).mean().to_numpy()
        vol_surge = (vtail - vol_mean) / vol_mean
        dev_score = (vwap - tail) / std

        trend = np.full_like(tail, np.nan, dtype=np.float64)
        w = 20; xs = np.arange(w)
        for i in range(w - 1, len(tail)):
            y = tail[i - w + 1:i + 1]
            if np.any(~np.isfinite(y)):
                continue
            trend[i] = np.polyfit(xs, y, 1)[0]

        F = np.vstack([z, dz, avg_dz, ma_slope, vol_surge, dev_score, trend]).T
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if len(F) < self.seq_len:
            return None
        return F[-self.seq_len:]

    def _realized_vol(self, buf: RollingBarBuffer) -> float:
        arr = buf.to_arrays()
        if not arr:
            return 0.0
        close = arr["close"]
        if len(close) < (self.vol_window + 2):
            return 0.0
        r = pd.Series(close).pct_change().dropna().to_numpy()
        if len(r) < 3:
            return 0.0
        tail = r[-self.vol_window:] if len(r) >= self.vol_window else r
        vol = float(np.std(tail))
        return vol if math.isfinite(vol) else 0.0

    def _market_returns(self) -> Tuple[float, float]:
        arr = self.buf_mkt.to_arrays()
        if not arr:
            return 0.0, 0.0
        c = arr["close"]
        if len(c) < 35:
            return 0.0, 0.0
        ret5 = (c[-1] / c[-6] - 1.0) if c[-6] else 0.0
        ret30 = (c[-1] / c[-31] - 1.0) if c[-31] else 0.0
        return float(ret5), float(ret30)

    # ──────────────────────────────────────────────────────────────────────
    # Signal prediction
    # ──────────────────────────────────────────────────────────────────────
    def predict_signal(self, buf: RollingBarBuffer) -> float:
        if self.model is None or self.scaler is None:
            return self._fallback_signal(buf)
        X_seq = self._compute_features_from_buffer(buf)
        if X_seq is None:
            return self._fallback_signal(buf)
        X2 = X_seq.reshape(-1, X_seq.shape[-1])
        X2s = self.scaler.transform(X2)
        Xs = X2s.reshape(1, self.seq_len, X_seq.shape[-1])
        pred = float(self.model.predict(Xs, verbose=0)[0, 0])
        return pred if math.isfinite(pred) else self._fallback_signal(buf)

    def _fallback_signal(self, buf: RollingBarBuffer) -> float:
        X_seq = self._compute_features_from_buffer(buf)
        if X_seq is None or len(X_seq) < 3:
            return 0.0
        z, dz, avg_dz = float(X_seq[-1, 0]), float(X_seq[-1, 1]), float(X_seq[-1, 2])
        ma_slope, vol_surge, trend = float(X_seq[-1, 3]), float(X_seq[-1, 4]), float(X_seq[-1, 6])
        mr = (-0.00025 * z) + (0.00010 * (-avg_dz))
        mom = (0.00008 * dz) + (0.00005 * ma_slope) + (0.00004 * trend)
        vol_boost = 1.0 + max(-0.25, min(0.25, 0.15 * vol_surge))
        pred = (mr + mom) * vol_boost
        return float(max(-0.02, min(0.02, pred))) if math.isfinite(pred) else 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Calibration / thresholds (per-symbol)
    # ──────────────────────────────────────────────────────────────────────
    def _signal_stats(self, ss: SymbolState) -> Tuple[float, float]:
        if len(ss.signal_history) < 20:
            return 0.0, 0.0
        arr = np.array(ss.signal_history, dtype=np.float32)
        return float(arr.mean()), float(arr.std())

    def _signal_percentile_threshold(self, ss: SymbolState) -> float:
        if len(ss.signal_history) < 30:
            return self.base_entry_threshold
        arr = np.array(ss.signal_history, dtype=np.float32)
        return float(np.quantile(arr, ss.signal_percentile))

    def _signal_z_values(self, ss: SymbolState) -> Tuple[float, float]:
        if len(ss.signal_history) < 30:
            return 0.0, 0.0
        arr = np.array(ss.signal_history, dtype=np.float32)
        mu, sd = float(arr.mean()), float(arr.std())
        if sd < 1e-8:
            return 0.0, 0.0
        return float((arr[-1] - mu) / sd), float((arr[-2] - mu) / sd)

    # ──────────────────────────────────────────────────────────────────────
    # Frictions / regime / liquidity (v14.2 adjusted bounds)
    # ──────────────────────────────────────────────────────────────────────
    def _spread_proxy(self, buf: RollingBarBuffer) -> float:
        arr = buf.to_arrays()
        if not arr or len(arr["close"]) == 0:
            return 0.0
        h, l, c = arr["high"][-1], arr["low"][-1], arr["close"][-1]
        return float((h - l) / c) if c else 0.0

    def _estimate_cost_buffer(self, price: float, vol: float, extended: bool = False) -> float:
        if extended:
            return float(min(0.00100, 0.00010 + 0.20 * max(vol, 0.0)))
        return float(min(0.00030, 0.00003 + 0.10 * max(vol, 0.0)))

    def _passes_liquidity_gate(self, buf: RollingBarBuffer, vol: float) -> Tuple[bool, str]:
        """v14.2 liquidity check -- uses dollar volume and relaxed IEX thresholds."""
        arr = buf.to_arrays()
        if not arr or len(arr["volume"]) == 0:
            return False, "no_data"
        bar_vol = float(arr["volume"][-1])
        bar_price = float(arr["close"][-1]) if arr["close"][-1] else 0.0
        dollar_vol = bar_vol * bar_price

        # Hard floor: need SOME activity
        if bar_vol < 50 and dollar_vol < 10_000:
            return False, f"dead_ticker(vol={int(bar_vol)},${int(dollar_vol)})"

        # Soft gate: only block if both low volume AND high volatility
        if bar_vol < self.min_volume_gate and vol > self.vol_hi_gate:
            return False, f"thin+volatile(vol={int(bar_vol)},rv={vol:.4f})"

        # Dollar volume floor (catches penny stocks with high share volume but no $)
        if dollar_vol < self.min_dollar_volume_gate and vol > self.vol_hi_gate:
            return False, f"low_dollar_vol(${int(dollar_vol)})"

        return True, "ok"

    def _regime_filter(self, now_et: datetime, vol: float, buf: RollingBarBuffer) -> Tuple[bool, str]:
        _, in_regular, is_extended_only, _ = self._session_flags(now_et)
        if in_regular:
            open_h, open_m = self.regular_open_hm
            close_h, close_m = self.regular_close_hm
        elif is_extended_only:
            open_h, open_m = self.extended_open_hm
            close_h, close_m = self.extended_close_hm
        else:
            return False, "session_closed"

        open_time = now_et.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        close_time = now_et.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        if now_et < open_time + timedelta(minutes=self.no_trade_open_minutes):
            return False, "open_window"
        if now_et > close_time - timedelta(minutes=self.no_trade_close_minutes):
            return False, "close_window"

        # Liquidity gate (v14.2)
        liq_ok, liq_reason = self._passes_liquidity_gate(buf, vol)
        if not liq_ok:
            return False, liq_reason

        # Extreme chop
        rng = self._spread_proxy(buf)
        if rng > (self.range_chop_gate * 2.5) and vol > (self.vol_hi_gate * 1.25):
            return False, f"chop(range={rng:.4f})"

        # Pause
        if self._paused_until and now_et < self._paused_until:
            return False, "paused"
        return True, "ok"

    # ──────────────────────────────────────────────────────────────────────
    # Sizing, orders
    # ──────────────────────────────────────────────────────────────────────
    def _compute_stop_take(self, vol: float) -> Tuple[float, float]:
        vol = max(vol, 1e-6)
        return float(-self.stop_k * vol), float(self.take_k * vol)

    def _risk_based_qty(self, equity: float, buying_power: float, price: float,
                        stop_pct: float, risk_per_trade: float) -> int:
        risk_dollars = equity * risk_per_trade
        stop_per_share = price * max(abs(stop_pct), 1e-6)
        shares_risk = int(risk_dollars / stop_per_share)
        notional_cap = buying_power * self.max_notional_fraction
        shares_cap = int(notional_cap / price) if price > 0 else 0
        return int(max(0, min(shares_risk, shares_cap, self.max_qty)))

    def _total_exposure(self, positions: Dict[str, Dict], equity: float) -> float:
        """Current total exposure as fraction of equity."""
        total = sum(abs(p["market_value"]) for p in positions.values())
        return total / max(equity, 1.0)

    def place_entry_bracket(self, symbol: str, qty: int, stop_pct: float, take_pct: float,
                            now_et: datetime, price: float) -> Optional[str]:
        _, _, is_extended_only, _ = self._session_flags(now_et)
        stop_price = price * (1.0 + stop_pct)
        take_price = price * (1.0 + take_pct)
        try:
            if is_extended_only:
                limit_px = round(price * 1.0015, 2)
                oreq = LimitOrderRequest(
                    symbol=symbol, qty=int(qty), side=OrderSide.BUY,
                    limit_price=float(limit_px), time_in_force=TimeInForce.DAY,
                    extended_hours=True)
            elif _BRACKET_OK:
                oreq = MarketOrderRequest(
                    symbol=symbol, qty=int(qty), side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(take_price, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_price, 2)))
            else:
                oreq = MarketOrderRequest(
                    symbol=symbol, qty=int(qty), side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY)
            order = self._call_trading_api(self.trading_client.submit_order, oreq)
            oid = getattr(order, "id", "") or ""
            print(f"    BUY {symbol}: qty={qty} (bracket={_BRACKET_OK and not is_extended_only})")
            return oid
        except Exception as e:
            print(f"    Entry failed {symbol}: {e}")
            return None

    def place_market_exit(self, symbol: str, qty: int, now_et: datetime) -> Optional[str]:
        try:
            _, _, is_extended_only, _ = self._session_flags(now_et)
            if is_extended_only:
                # Need a price reference for limit order
                ss = self.sym_states.get(symbol)
                price = ss.buf.last_close() if ss else None
                if not price:
                    return None
                oreq = LimitOrderRequest(
                    symbol=symbol, qty=int(qty), side=OrderSide.SELL,
                    limit_price=round(price * 0.9985, 2),
                    time_in_force=TimeInForce.DAY, extended_hours=True)
            else:
                oreq = MarketOrderRequest(
                    symbol=symbol, qty=int(qty), side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY)
            order = self._call_trading_api(self.trading_client.submit_order, oreq)
            oid = getattr(order, "id", "") or ""
            print(f"    SELL {symbol}: qty={qty}")
            return oid
        except Exception as e:
            print(f"    Exit failed {symbol}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Entry confirmation (per-symbol)
    # ──────────────────────────────────────────────────────────────────────
    def _entry_confirmed(self, signal: float, vol: float, now_et: datetime,
                         ss: SymbolState) -> Tuple[bool, str, float]:
        price = ss.buf.last_close()
        if price is None:
            return False, "no_price", 0.0

        _, _, is_extended_only, _ = self._session_flags(now_et)
        cost_buf = self._estimate_cost_buffer(price, vol, extended=is_extended_only)

        # Dynamic threshold
        mu, sd = self._signal_stats(ss)
        thr_dyn = max(self.base_entry_threshold, mu + self.entry_zscore_sigma * sd)
        thr_pct = self._signal_percentile_threshold(ss)
        sig_z, sig_z_prev = self._signal_z_values(ss)

        # High-vol penalty
        hi_vol_penalty = (0.00010 + 0.25 * cost_buf) if vol > self.vol_hi_gate else 0.0
        thr_final = max(thr_dyn, thr_pct) + hi_vol_penalty

        # Range penalty
        rng = self._spread_proxy(ss.buf)
        if rng > self.range_chop_gate:
            thr_final += 0.00008

        # Tape confirmation
        mkt5, _ = self._market_returns()
        if mkt5 < self.tape_long_min_5m:
            return False, f"tape_block(mkt5={mkt5:+.3%})", cost_buf

        # Edge over cost
        if signal <= (cost_buf + self.min_edge_over_cost):
            return False, "edge<cost", cost_buf

        # Signal z-score check
        z_ok = (sig_z >= self.entry_sig_z_main or
                (sig_z >= self.entry_sig_z_persist and sig_z_prev >= self.entry_sig_z_persist) or
                sig_z >= self.entry_sig_z_override)
        if not z_ok:
            return False, f"sigz({sig_z:.2f})", cost_buf

        if signal <= thr_final:
            return False, f"below_thr({signal:.6f}<{thr_final:.6f})", cost_buf

        if ss.cooldown_bars_remaining > 0:
            return False, "cooldown", cost_buf

        return True, "ok", cost_buf

    # ──────────────────────────────────────────────────────────────────────
    # Online evaluation (per-symbol adaptive)
    # ──────────────────────────────────────────────────────────────────────
    def _update_online_eval(self, ss: SymbolState, pnl_pct: float):
        ss.recent_trade_pnls.append(float(pnl_pct))
        ss.recent_wins.append(1 if pnl_pct > 0 else 0)
        if len(ss.recent_trade_pnls) >= 10:
            win_rate = sum(ss.recent_wins) / len(ss.recent_wins)
            avg = float(np.mean(ss.recent_trade_pnls))
            if win_rate < 0.45 and avg < 0:
                ss.signal_percentile = min(0.85, ss.signal_percentile + 0.02)
                ss.risk_per_trade = max(0.0010, ss.risk_per_trade * 0.9)
            elif win_rate > 0.55 and avg > 0:
                ss.signal_percentile = max(0.60, ss.signal_percentile - 0.01)
                ss.risk_per_trade = min(0.0030, ss.risk_per_trade * 1.05)

    # ──────────────────────────────────────────────────────────────────────
    # Session / hygiene helpers
    # ──────────────────────────────────────────────────────────────────────
    def _session_flags(self, now_et: datetime) -> Tuple[bool, bool, bool, str]:
        if now_et.weekday() >= 5:
            return False, False, False, "weekend"
        reg_open = now_et.replace(hour=self.regular_open_hm[0], minute=self.regular_open_hm[1], second=0, microsecond=0)
        reg_close = now_et.replace(hour=self.regular_close_hm[0], minute=self.regular_close_hm[1], second=0, microsecond=0)
        ext_open = now_et.replace(hour=self.extended_open_hm[0], minute=self.extended_open_hm[1], second=0, microsecond=0)
        ext_close = now_et.replace(hour=self.extended_close_hm[0], minute=self.extended_close_hm[1], second=0, microsecond=0)
        in_regular = reg_open <= now_et <= reg_close
        in_extended = self.enable_extended_hours and ext_open <= now_et <= ext_close
        is_open = in_regular or in_extended
        is_extended_only = in_extended and not in_regular
        label = "regular" if in_regular else ("extended" if is_extended_only else "closed")
        return is_open, in_regular, is_extended_only, label

    def _reset_daily_if_needed(self, now_et: datetime):
        d = now_et.date()
        if self._today != d:
            self._today = d
            self._trades_today = 0
            acct = self.get_account_info(force=True)
            self._day_start_equity = float(acct["equity"])
            self._paused_until = None
            print(f"\n  -- New day: {d.isoformat()} | equity=${self._day_start_equity:,.2f}")

    def _daily_kill_switch_check(self) -> Tuple[bool, str]:
        if self._day_start_equity is None:
            return True, "no_baseline"
        acct = self.get_account_info()
        dd = (float(acct["equity"]) / self._day_start_equity) - 1.0
        if dd <= -self.daily_max_drawdown:
            self._paused_until = datetime.now(self.eastern) + timedelta(minutes=self.pause_minutes_on_kill)
            return False, f"kill_dd={dd:.2%}"
        return True, "ok"

    # ──────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────
    def _init_logs(self):
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "ts", "symbol", "side", "qty", "price_ref",
                    "signal", "vol", "cost_buf", "stop_pct", "take_pct", "reason", "order_id"])
        if not self.state_log_path.exists():
            with open(self.state_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "ts", "symbol", "price", "signal", "vol", "pos_qty", "pos_pnl_pct",
                    "held_bars", "trades_today"])

    def _log_trade(self, symbol: str, side: str, qty: int, price_ref: float,
                   signal: float, vol: float, cost_buf: float,
                   stop_pct: float, take_pct: float, reason: str, order_id: str = ""):
        with open(self.trade_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(self.eastern).isoformat(), symbol, side, qty, f"{price_ref:.2f}",
                f"{signal:.6f}", f"{vol:.5f}", f"{cost_buf:.6f}",
                f"{stop_pct:.4f}", f"{take_pct:.4f}", reason, order_id])

    # ══════════════════════════════════════════════════════════════════════
    # SIGNAL ENGINE ADAPTER (bridges v14.2 signal model -> v15 TradeSignal)
    # ══════════════════════════════════════════════════════════════════════
    def generate_signals(self) -> list:
        """
        Generate TradeSignal objects for all symbols with sufficient data.
        Called by the execution loop each cycle AFTER broker truth is established.
        """
        from execution_controller import (
            TradeSignal, MarketState, CostEstimate, compute_setup_fingerprint
        )

        signals = []
        now = datetime.now(self.eastern)

        for sym, ss in self.sym_states.items():
            if len(ss.buf) < self.lookback + self.seq_len:
                continue

            # Compute raw signal
            raw_signal = self.predict_signal(ss.buf)
            ss.bars_seen += 1
            signal_delta = 0.0 if ss.last_signal is None else (raw_signal - ss.last_signal)
            ss.last_signal = raw_signal
            ss.signal_history.append(raw_signal)

            # Tick cooldown
            if ss.cooldown_bars_remaining > 0:
                ss.cooldown_bars_remaining -= 1

            vol = self._realized_vol(ss.buf)
            vol = max(vol, 1e-6)
            price = ss.buf.last_close() or 0.0
            if price <= 0:
                continue

            # Regime filter (skip dead/choppy tickers at signal level)
            ok_regime, _ = self._regime_filter(now, vol, ss.buf)

            # Determine direction based on raw signal strength.
            # The signal engine does NOT gate entries -- that's the execution controller's job.
            # We just classify the signal as "long" or "flat" based on whether the model
            # sees positive expected return.
            if not ok_regime:
                # Dead/choppy ticker -> always flat
                direction = "flat"
            elif raw_signal > 0:
                # Positive expected return -> propose long
                direction = "long"
            else:
                # Negative -> propose flat (exit if holding)
                direction = "flat"

            # Expected return in bps (signal is in return-fraction units)
            expected_return_bps = raw_signal * 10000.0

            # Setup fingerprint
            sig_z, _ = self._signal_z_values(ss)
            vol_regime = "high" if vol > self.vol_hi_gate else "normal"
            trend_state = "up" if raw_signal > 0 else "down"
            fingerprint = compute_setup_fingerprint(
                sym, "B4" if raw_signal > 0.0003 else "B3",
                direction, sig_z, vol_regime, trend_state)

            # Bucket assignment based on signal strength
            if raw_signal > 0.0005:
                bucket_id = "B5"
            elif raw_signal > 0.0003:
                bucket_id = "B4"
            elif raw_signal > 0.0001:
                bucket_id = "B3"
            else:
                bucket_id = "B2"

            # Invalidation price (stop level)
            stop_pct, _ = self._compute_stop_take(vol)
            invalidation_price = price * (1.0 + stop_pct) if direction == "long" else None

            # TTL based on bucket
            ttl_map = {"B5": 60, "B4": 75, "B3": 90, "B2": 120}
            ttl = ttl_map.get(bucket_id, 90)

            ts = TradeSignal(
                symbol=sym,
                bucket_id=bucket_id,
                direction=direction,
                score=min(1.0, max(0.0, (raw_signal + 0.001) / 0.002)),  # normalize to 0-1
                expected_return_bps=expected_return_bps,
                setup_fingerprint=fingerprint,
                generated_at=datetime.utcnow(),
                ttl_seconds=ttl,
                invalidation_price=invalidation_price,
                reason=f"sig={raw_signal:.6f}|vol={vol:.5f}|z={sig_z:.2f}",
            )
            signals.append(ts)

        return signals

    def get_market_state(self, symbol: str):
        """Build MarketState for a symbol (used by execution controller)."""
        from execution_controller import MarketState

        ss = self.sym_states.get(symbol)
        if ss is None:
            return MarketState()

        vol = self._realized_vol(ss.buf)
        price = ss.buf.last_close() or 0.0

        # ATR proxy from vol (vol is per-minute std of returns)
        atr = price * vol * 2.5 if price > 0 else 0.0

        # Structure reset: considered reset after enough bars pass
        # (simplified: uses bars_seen which resets daily; good enough for fresh campaigns)
        structure_reset = True  # default to True for new campaigns with no exit history

        # Continuation: signal still positive and trending
        continuation = (ss.last_signal is not None and ss.last_signal > self.base_entry_threshold)

        return MarketState(
            bars_since_last_exit=999,  # no exit history for fresh campaigns
            structure_reset=structure_reset,
            continuation_confirmed=continuation,
            atr=atr,
            last_price=price,
        )

    def estimate_costs(self, symbol: str):
        """Build CostEstimate for a symbol."""
        from execution_controller import CostEstimate

        ss = self.sym_states.get(symbol)
        if ss is None:
            return CostEstimate()

        vol = self._realized_vol(ss.buf)
        spread = self._spread_proxy(ss.buf) * 10000.0  # convert to bps

        # Slippage scales with vol
        slippage = max(1.0, vol * 5000.0)

        return CostEstimate(
            spread_bps=max(1.0, spread),
            slippage_bps=slippage,
            churn_penalty_bps=0.0,
            stale_penalty_bps=0.0,
        )
    # ══════════════════════════════════════════════════════════════════════
    # RUN (V14_2_1 Loss-Governed Active Portfolio Controller)
    # ══════════════════════════════════════════════════════════════════════
    def run(self, check_interval=45):
        """
        Main entry point. Delegates to run_controller_loop from execution_controller.py.
        Signal model stays unchanged -- only operations changed.
        """
        from execution_controller import (
            ExecutionPolicy, CampaignBook, AlpacaBrokerAdapter, run_controller_loop,
        )

        policy = ExecutionPolicy(
            allow_extended_hours=False,
            no_new_entries_before="09:35",
            no_new_entries_after="15:55",
            cancel_premarket_orders_at="09:29",
            max_risk_per_trade_pct=self.base_risk_per_trade,
            max_symbol_notional_pct=self.max_notional_fraction,
            max_shares_per_symbol=self.max_qty,
            max_total_exposure_pct=self.max_total_exposure,
            max_concurrent_positions=self.max_concurrent_positions,
            max_trades_per_day=self.max_trades_per_day,
            daily_max_drawdown_pct=self.daily_max_drawdown,
            min_net_edge_bps=1.5,
        )

        broker_adapter = AlpacaBrokerAdapter(self.trading_client, self.eastern)
        campaign_book = CampaignBook(self.symbols)
        campaign_book.policy_ref = policy

        # Bootstrap signal buffers before entering the controller loop
        self.bootstrap_buffers()

        run_controller_loop(
            signal_engine=self,
            broker_adapter=broker_adapter,
            campaign_book=campaign_book,
            policy=policy,
            eastern_tz=self.eastern,
            check_interval=check_interval,
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Daily run:  python alpaca_trader.py
    Options:    python alpaca_trader.py --symbols TSLA,NVDA,AMD --interval 30
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Alpaca V14.2.1 Loss-Governed Active Portfolio Controller")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated ticker list (default: full volatile universe)")
    parser.add_argument("--interval", type=int, default=45,
                        help="Seconds between polling cycles (default: 45)")
    parser.add_argument("--model", type=str, default="HFT/model/instinct_model.weights.h5",
                        help="Path to model weights")
    parser.add_argument("--scaler", type=str, default="HFT/model/scaler.pkl",
                        help="Path to scaler pickle")
    parser.add_argument("--max-positions", type=int, default=8,
                        help="Max concurrent positions (default: 8)")
    parser.add_argument("--max-exposure", type=float, default=0.85,
                        help="Max portfolio exposure fraction (default: 0.85)")
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    trader = AlpacaTrader(
        symbols=symbols,
        model_path=args.model,
        scaler_path=args.scaler,
    )
    trader.max_concurrent_positions = args.max_positions
    trader.max_total_exposure = args.max_exposure
    trader.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
