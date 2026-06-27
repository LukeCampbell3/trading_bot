# crypto_trader.py – Binance API crypto trader with model + pattern voting
# ----------------------------------------------------------------------------
import time, requests, joblib, numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# ─── Configuration ───────────────────────────────────────────────────────────
API_KEY = ""
API_SECRET = ""
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LOOKBACK = 100

MODEL_PATH  = "model/crypto_model.keras"
SCALER_PATH = "model/crypto_scaler.pkl"

BUY_THRESH   = 0.0010
TREND_MIN    = -0.15
POS_FRAC     = 0.90
STOP_LOSS_P  = 0.015
TAKE_PROFIT  = 0.012
COOLDOWN     = 3

# ─── Binance API ----------------------------------------------------------------
class BinanceAPI:
    BASE = "https://api.binance.com"

    def get_recent_data(self, symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK+5):
        url = f"{self.BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        res = requests.get(url, params=params)
        data = res.json()
        prices  = np.array([float(x[4]) for x in data], dtype=np.float32)
        volumes = np.array([float(x[5]) for x in data], dtype=np.float32)
        return prices, volumes

    def get_balance(self, asset):
        # Placeholder
        return 1000.0 if asset == "USDT" else 0.0

    def place_order(self, side, qty):
        print(f"[{datetime.now()}] {side.upper()} {qty:.6f}")
        return True

# ─── Pattern Detection ───────────────────────────────────────────────────────
def detect_inverse_head_shoulders(prices):
    if len(prices) < 20: return False, 0.0
    p = prices[-20:]
    l = len(p)
    left  = np.min(p[:l//3])
    head  = np.min(p[l//3:l*2//3])
    right = np.min(p[l*2//3:])
    valid = (head < left and head < right) and abs(left - right)/head < 0.1
    confidence = 1.0 - abs(left - right)/max(left, right) if valid else 0.0
    return valid, confidence

def detect_double_bottom(prices):
    if len(prices) < 20: return False, 0.0
    p = prices[-20:]
    troughs = sorted([p[i] for i in range(1, len(p)-1) if p[i] < p[i-1] and p[i] < p[i+1]])
    if len(troughs) < 2: return False, 0.0
    valid = abs(troughs[-1] - troughs[-2])/troughs[-1] < 0.05
    confidence = 1.0 - abs(troughs[-1] - troughs[-2])/max(troughs[-1], troughs[-2]) if valid else 0.0
    return valid, confidence

# ─── Predictor & Voting ──────────────────────────────────────────────────────
class Predictor:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def compute_features(self, prices, volumes):
        ma20  = np.mean(prices[-20:])
        ma30  = np.mean(prices[-30:])
        std   = np.std(prices[-30:])
        vwap  = np.average(prices[-LOOKBACK:], weights=volumes[-LOOKBACK:])
        z     = (prices[-1] - vwap)/std
        dz    = z - (prices[-2] - vwap)/std
        trend = np.polyfit(range(20), prices[-20:], 1)[0]
        return np.array([z, dz, ma20-ma30, trend], dtype=np.float32)

    def predict(self, prices, volumes):
        feats = self.compute_features(prices, volumes).reshape(1, -1)
        feats = self.scaler.transform(feats)
        return float(self.model.predict(feats, verbose=0)[0, 0]), feats

    def vote(self, prices, volumes):
        score, _ = self.predict(prices, volumes)
        votes = []

        if score > BUY_THRESH:
            votes.append(("model", score))

        if np.polyfit(range(20), prices[-20:], 1)[0] > TREND_MIN:
            votes.append(("trend", 0.8))

        hns, hns_conf = detect_inverse_head_shoulders(prices)
        if hns: votes.append(("inv_h&s", hns_conf))

        dbl, dbl_conf = detect_double_bottom(prices)
        if dbl: votes.append(("dbl_bottom", dbl_conf))

        if not votes: return 0.0, []
        weighted = sum([v[1] for v in votes]) / len(votes)
        return weighted, votes

# ─── Trader Engine ───────────────────────────────────────────────────────────
def run():
    api  = BinanceAPI()
    pred = Predictor()
    cool = 0
    entry_price = 0.0

    while True:
        prices, vols = api.get_recent_data()
        if len(prices) < LOOKBACK: continue

        signal, votes = pred.vote(prices, vols)
        price = prices[-1]
        pos   = api.get_balance("BTC") > 0.0001

        print(f"[{datetime.now():%H:%M:%S}] signal={signal:.4f} pos={pos} price={price:.2f} votes={votes}")

        if cool > 0:
            cool -= 1
            time.sleep(15)
            continue

        if not pos and signal > BUY_THRESH:
            usdt = api.get_balance("USDT")
            qty  = (usdt * POS_FRAC) / price
            if qty > 0.0001:
                api.place_order("buy", qty)
                entry_price = price

        elif pos:
            change = (price - entry_price)/entry_price
            if change < -STOP_LOSS_P or change > TAKE_PROFIT:
                qty = api.get_balance("BTC")
                api.place_order("sell", qty)
                cool = COOLDOWN

        time.sleep(15)

if __name__ == "__main__":
    run()
