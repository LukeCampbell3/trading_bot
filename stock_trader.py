# adaptive_strategy_live_robust.py
#
# – Shape-safe feature builder
# – Updated Alpaca-py client usage
# – NaN / Inf guard on every input that touches the model
# – Optional bar-interval so you can shorten the sleep if you want

import argparse, joblib, numpy as np, pandas as pd, time as systime
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dttime
import pytz, collections, warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests    import StockBarsRequest
from alpaca.data.timeframe   import TimeFrame
from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums    import OrderSide, TimeInForce

# ─── USER CONFIG ─────────────────────────────────────────────────────────────
ALPACA_KEY     = "PKVPJ9MCE23AS06OSXP7"                			           # ←  your key
ALPACA_SECRET  = "Ojy3hoWuX1tTfNn15Hu0H06l0Lttz4cS1pgpPtqi"                # ←  your secret
PAPER          = True               # flip to False for live
TICKER         = "PLTR"
MODEL_P        = Path(r"C:/Users/jcthi/Code/HFT/model/instinct_model.keras")
SCALER_P       = Path(r"C:/Users/jcthi/Code/HFT/model/scaler.pkl")
BAR_MULTIPLIER = 30
SLEEP_SEC = BAR_MULTIPLIER * 60

@dataclass
class HParams:
    lookback       : int   = 120
    buy_thresh     : float = 0.00018
    trend_min      : float = 0.001
    hold_bars      : int   = 3
    pos_frac_max   : float = 0.40
    stop_base      : float = 0.05
    take_base      : float = 0.015
    atr_win        : int   = 14
    rsi_win        : int   = 14
    macd_fast      : int   = 12
    macd_slow      : int   = 26
    macd_signal    : int   = 9
    cooldown_bars  : int   = 2
    rl_window      : int   = 20
HP = HParams()

# ─── Predictor ───────────────────────────────────────────────────────────────
class Predictor:
    def __init__(self):
        self.model  = load_model(MODEL_P, compile=False)
        self.scaler = joblib.load(SCALER_P)

    def _basic_feats(self, prices: np.ndarray, vols: np.ndarray) -> np.ndarray:
        """7 robust, NaN-free features for the ML model."""
        h = HP
        if len(prices) < h.lookback:        # guard – never let shape mismatch through
            raise ValueError(f"Need ≥{h.lookback} bars, got {len(prices)}")

        wp = prices[-h.lookback:]
        wv = vols[-h.lookback:]
        vol_sum = max(wv.sum(), 1e-9)
        vwap    = float(np.dot(wp, wv) / vol_sum)

        std  = max(wp.std(ddof=0), 1e-6)
        z_now, z_prev = (wp[-1]-vwap)/std, (wp[-2]-vwap)/std
        dz     = z_now - z_prev
        slope  = (wp[-1] - wp[0]) / h.lookback
        vol_s  = (wv[-1] / max(wv[-10:].mean(), 1e-6)) - 1.0
        dev_s  = (vwap - wp[-1]) / std
        trend  = np.polyfit(np.arange(h.lookback), wp, 1)[0]

        feats  = np.array([z_now, dz, dz, slope, vol_s, dev_s, trend], np.float32)
        if np.any(~np.isfinite(feats)):
            warnings.warn("Non-finite features detected – replaced with 0.")
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    def predict(self, prices: np.ndarray, vols: np.ndarray) -> float:
        feats   = self._basic_feats(prices, vols)
        scaled  = self.scaler.transform(feats.reshape(1, -1))
        return float(self.model.predict(scaled, verbose=0)[0, 0])

# ─── Indicator Helpers (unchanged) ───────────────────────────────────────────
def macd_hist(series: pd.Series) -> float:
    fast   = series.ewm(span=HP.macd_fast).mean()
    slow   = series.ewm(span=HP.macd_slow).mean()
    macd   = fast - slow
    signal = macd.ewm(span=HP.macd_signal).mean()
    return float((macd - signal).iloc[-1])

def rsi(series: pd.Series, period: int) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period).mean()
    loss  = -delta.clip(upper=0).ewm(alpha=1/period).mean()
    rs    = gain / (loss + 1e-6)
    return float(100 - 100/(1+rs).iloc[-1])

def atr(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    tr  = pd.concat([high-low,
                     (high-close.shift()).abs(),
                     (low-close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(HP.atr_win).mean().iloc[-1])

# ─── Alpaca Broker Wrapper ───────────────────────────────────────────────────
class AlpacaBroker:
    def __init__(self):
        self.client = TradingClient(
            ALPACA_KEY, ALPACA_SECRET,
            paper=PAPER,  # current SDK still accepts this flag :contentReference[oaicite:0]{index=0}
        )
        self.data   = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    # ---- Market-data ----
    def bars(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        now   = datetime.now(pytz.UTC)
        start = now - timedelta(minutes=limit*BAR_INTERVAL.n + 60)
        req   = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=BAR_INTERVAL,
            start=start, end=now, limit=limit
        )
        raw   = self.data.get_stock_bars(req).df
        if raw.empty:
            return pd.DataFrame()
        df = (raw
              .xs(symbol, level="symbol")
              .reset_index()
              .rename(columns={
                  "timestamp": "ts",
                  "open":  "open",
                  "high":  "high",
                  "low":   "low",
                  "close": "price",
                  "volume": "volume"
              })
        )
        return df

    # ---- Account helpers ----
    def _safe(self, fn, fallback=np.nan):
        try:
            return fn()
        except Exception as e:
            print("[WARN]", e)
            return fallback

    def account(self): return self.client.get_account()
    def equity       (self) -> float:   return float(self.account.equity)
    def buying_power (self) -> float:   return float(self.account.buying_power)
    def position_qty (self) -> int:
        pos = self._safe(lambda: self.client.get_open_position(TICKER))
        return int(pos.qty) if pos else 0
    def entry_price  (self) -> float:
        pos = self._safe(lambda: self.client.get_open_position(TICKER))
        return float(pos.avg_entry_price) if pos else 0

    # ---- Trading helpers ----
    def buy(self, qty: int, extended: bool):
        print(f"[{datetime.now()}] BUY {qty}")
        self._safe(lambda: self.client.submit_order(
            MarketOrderRequest(
                symbol=TICKER, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                extended_hours=extended
            )
        ))

    def sell_all(self, extended: bool):
        qty = self.position_qty()
        if qty:
            print(f"[{datetime.now()}] SELL {qty}")
            self._safe(lambda: self.client.submit_order(
                MarketOrderRequest(
                    symbol=TICKER, qty=qty, side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=extended
                )
            ))

# ─── Reinforcement Memory (unchanged) ────────────────────────────────────────
Trade = collections.namedtuple("Trade", "pnl")
class RLMemory:
    def __init__(self, window=HP.rl_window):
        self.window, self.trades = window, []
    def add(self, pnl: float):
        self.trades.append(pnl)
        if len(self.trades) > self.window:
            self.trades.pop(0)
    def win_rate(self):
        if not self.trades: return 0.5
        return sum(p > 0 for p in self.trades) / len(self.trades)

# ─── Main Trading Engine (logic identical, but shape-safe) ───────────────────
def run_live(extended: bool):
    session = ("04:00", "20:00") if extended else ("09:30", "16:00")
    open_t, close_t = map(dttime.fromisoformat, session)

    broker, ai, mem = AlpacaBroker(), Predictor(), RLMemory()
    eastern         = pytz.timezone("US/Eastern")
    cooldown        = 0

    print(f"Waiting for session {open_t}–{close_t} ET…")
    while True:
        now_et = datetime.now(eastern)

        if not (open_t <= now_et.time() <= close_t):
            systime.sleep(300)         # sleep 5 min outside market hours
            continue

        bars = broker.bars(TICKER, limit=max(HP.lookback + HP.hold_bars, 200))
        if len(bars) < HP.lookback + HP.hold_bars:
            print("⤻ Need more bars – retry in 15 min")
            systime.sleep(900)
            continue

        prices, vols = bars.price.values.astype(np.float32), bars.volume.values.astype(np.float32)

        # — Indicators — ----------------------------------------------------
        macd_h  = macd_hist(bars.price)
        rsi_val = rsi(bars.price, HP.rsi_win)
        atr_val = atr(bars.high, bars.low, bars.price)
        vwap    = (bars.price * bars.volume).cumsum() / bars.volume.cumsum()
        trend   = np.polyfit(np.arange(HP.lookback), prices[-HP.lookback:], 1)[0]

        # — ML prediction — -------------------------------------------------
        try:
            pred_now = ai.predict(prices, vols)
        except ValueError as ve:       # any shape issue ends the loop safely
            print("[WARN]", ve)
            systime.sleep(SLEEP_SEC)
            continue

        # — Voting & Persistence — -----------------------------------------
        votes = [
            pred_now > HP.buy_thresh,
            trend    > HP.trend_min,
            macd_h   > 0,
            rsi_val  < 70,
            prices[-1] < vwap.iloc[-1],
        ]
        score = sum(votes)

        persistent = score >= 3
        if persistent:
            for i in range(-HP.hold_bars, 0):
                window_ok = len(prices[:i]) >= HP.lookback
                if not window_ok: 
                    persistent = False
                    break

                p = ai.predict(prices[:i], vols[:i])
                t = np.polyfit(np.arange(HP.lookback), prices[i-HP.lookback:i], 1)[0]
                if not (p > HP.buy_thresh and t > HP.trend_min):
                    persistent = False
                    break

        # — Portfolio state — ----------------------------------------------
        pos_qty     = broker.position_qty()
        price       = prices[-1]
        entry_price = broker.entry_price() if pos_qty else 0
        move        = (price - entry_price) / entry_price if pos_qty else 0

        # RL dynamic threshold
        buy_thresh_dyn = HP.buy_thresh * (1.2 if mem.win_rate() < 0.45 else 0.9)

        if cooldown: cooldown -= 1

        # — ENTRY — ---------------------------------------------------------
        if not pos_qty and not cooldown and persistent and pred_now > buy_thresh_dyn:
            confidence = min(1.0, (pred_now - buy_thresh_dyn) * 5)
            qty = int(broker.buying_power() * confidence * HP.pos_frac_max // price)
            if qty > 0:
                broker.buy(qty, extended)

        # — EXIT — ----------------------------------------------------------
        if pos_qty:
            reverse_votes = [
                pred_now < HP.buy_thresh,
                macd_h   < 0,
                rsi_val  > 70,
                prices[-1] > vwap.iloc[-1],
            ]
            reverse  = sum(reverse_votes) >= 3
            stop_pct = HP.stop_base * (1 + atr_val / price)
            take_pct = HP.take_base * (1 + atr_val / price)

            if move < -stop_pct or move > take_pct or reverse:
                broker.sell_all(extended)
                mem.add(move)
                cooldown = HP.cooldown_bars

        print(f"[{now_et:%Y-%m-%d %H:%M}]"
              f" p={price:.2f} s={score} pred={pred_now:.4f} tr={trend:.4f}"
              f" rsi={rsi_val:5.1f} macd={macd_h:.4f} atr={atr_val:.4f}"
              f" pos={pos_qty} move={move:+.2%} winr={mem.win_rate():.2}")

        systime.sleep(SLEEP_SEC)       # wait exactly one bar

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live",  action="store_true", help="start trading")
    ap.add_argument("--after", action="store_true", help="include extended hours")
    args = ap.parse_args()

    if args.live:
        run_live(extended=args.after)
    else:
        print("Run with --live (optionally --after) to start the trader.")
