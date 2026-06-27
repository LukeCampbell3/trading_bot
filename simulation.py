import pandas as pd
import cupy as cp
import numpy as np
import numba
import threading
import time
import tensorflow as tf
import os
from glob import glob
from tensorflow.keras.models import load_model
from transformer import WarmupThenDecay

class DataHandler:
    def __init__(self, folder_path, chunk_size=100):
        self.feather_files = sorted(glob(os.path.join(folder_path, "*.feather")))
        self.chunk_size = chunk_size
        self.current_file_index = 0
        self.data = pd.DataFrame()
        self.index = 0
        self._load_next_file()

    def _load_next_file(self):
        if self.current_file_index < len(self.feather_files):
            self.data = pd.read_feather(self.feather_files[self.current_file_index])
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.sort_values('timestamp', inplace=True)
            self.total_rows = len(self.data)
            self.index = 0
            self.current_file_index += 1
        else:
            self.data = pd.DataFrame()
            self.total_rows = 0

    def stream_data(self):
        while True:
            if self.index >= self.total_rows:
                self._load_next_file()
                if self.total_rows == 0:
                    break
            chunk = self.data.iloc[self.index:self.index + self.chunk_size]
            self.index += self.chunk_size
            yield chunk

class GPUPredictor:
    def predict(self, prices: cp.ndarray, volumes: cp.ndarray) -> float:
        if prices.size < 2 or volumes.sum() == 0:
            return 0.0
        vwap = cp.sum(prices * volumes) / cp.sum(volumes)
        std = cp.std(prices)
        if std == 0:
            return 0.0
        return float((prices[-1] - vwap) / std)

@numba.njit
def compute_zscore_cpu(prices: np.ndarray, volumes: np.ndarray) -> float:
    if prices.shape[0] < 2 or np.sum(volumes) == 0:
        return 0.0
    vwap = np.sum(prices * volumes) / np.sum(volumes)
    mean = np.mean(prices)
    std = np.sqrt(np.mean((prices - mean) ** 2))
    if std == 0:
        return 0.0
    return (prices[-1] - vwap) / std

class DecisionEngine:
    def __init__(self, base_threshold=1.5, min_threshold=0.5):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold

    def decide(self, combined_score: float, prices: list):
        volatility = np.std(prices[-30:])
        dynamic_threshold = max(self.base_threshold * volatility, self.min_threshold)
        if combined_score > dynamic_threshold:
            return "BUY"
        elif combined_score < -dynamic_threshold:
            return "SELL"
        return "SIT"

class ExecutionEngine:
    def __init__(self, initial_capital=10000):
        self.cash = initial_capital
        self.position = 0
        self.last_price = None
        self.trade_count = 0
        self.timestamps_seen = set()

    def execute(self, action, price, timestamp, size=1, slippage=0.001):
        if timestamp in self.timestamps_seen:
            return
        self.timestamps_seen.add(timestamp)

        effective_price = price * (1 + slippage if action == "BUY" else 1 - slippage)

        for _ in range(size):
            if action == "BUY" and self.cash >= effective_price:
                self.position += 1
                self.cash -= effective_price
                self.trade_count += 1
            elif action == "SELL" and self.position > 0:
                self.position -= 1
                self.cash += effective_price
                self.trade_count += 1

        self.last_price = price

    def final_report(self):
        portfolio_value = self.cash + (self.position * self.last_price if self.last_price else 0)
        print(f"\nPortfolio Value: ${portfolio_value:.2f} | Trades: {self.trade_count}")
        return portfolio_value

class Simulation:
    def __init__(self, feather_folder, model_path, symbol="STOCK", z_weight=0.5, model_weight=0.5, use_gpu=True, n_steps_ahead=3):
        self.symbol = symbol
        self.data_handler = DataHandler(feather_folder)
        self.use_gpu = use_gpu
        self.predictor = GPUPredictor()
        self.decision_engine = DecisionEngine()
        self.execution_engine = ExecutionEngine()
        self.model = load_model(model_path)
        self.sequence_length = self.model.input_shape[1]
        self.n_steps_ahead = n_steps_ahead
        self.z_weight = z_weight
        self.model_weight = model_weight
        self.prices = []
        self.volumes = []
        self.recent_trades = []
        self.prediction_cache = {}
        self.optimize_interval = 500
        self.hit_rate_threshold = 0.5
        self.lock = threading.Lock()
        threading.Thread(target=self.optimize_weights_loop, daemon=True).start()

    def optimize_weights_loop(self):
        while True:
            time.sleep(1)
            if len(self.recent_trades) >= self.optimize_interval:
                scores = self.recent_trades[-self.optimize_interval:]
                best_score = float('-inf')
                best_weights = (self.z_weight, self.model_weight)

                hit_rates = []
                for z, m, result in scores:
                    key = (round(z, 2), round(m, 2))
                    cache = self.prediction_cache.get(key, {'hits': 1, 'misses': 1})
                    hit_rate = cache['hits'] / (cache['hits'] + cache['misses'])
                    hit_rates.append(hit_rate)

                if hit_rates:
                    self.hit_rate_threshold = np.percentile(hit_rates, 60)

                for z_w in np.arange(0.0, 1.1, 0.1):
                    m_w = 1.0 - z_w
                    pnl = 0
                    for z, m, result in scores:
                        combined = z * z_w + m * m_w
                        pnl += result * np.sign(combined)
                    if pnl > best_score:
                        best_score = pnl
                        best_weights = (z_w, m_w)

                with self.lock:
                    self.z_weight, self.model_weight = best_weights

    def update_cache(self, z, m, outcome):
        key = (round(z, 2), round(m, 2))
        if key not in self.prediction_cache:
            self.prediction_cache[key] = {'hits': 0, 'misses': 0}
        if outcome > 0:
            self.prediction_cache[key]['hits'] += 1
        else:
            self.prediction_cache[key]['misses'] += 1

    def filter_trade(self, z, m):
        key = (round(z, 2), round(m, 2))
        cache = self.prediction_cache.get(key, {'hits': 1, 'misses': 1})
        hit_rate = cache['hits'] / (cache['hits'] + cache['misses'])
        return hit_rate > self.hit_rate_threshold

    def run(self, return_value=False):
        for chunk in self.data_handler.stream_data():
            try:
                if 'target' not in chunk.columns:
                    continue

                prices_chunk = chunk['target'].values.astype(np.float32)  # Using target as proxy for price
                volumes_chunk = np.ones_like(prices_chunk, dtype=np.float32)  # Dummy volume if not available
                timestamps = chunk['timestamp'].values

                self.prices.extend(prices_chunk.tolist())
                self.volumes.extend(volumes_chunk.tolist())

                if len(self.prices) < self.sequence_length:
                    continue

                recent_prices = np.array(self.prices[-self.sequence_length:], dtype=np.float32)
                recent_volumes = np.array(self.volumes[-self.sequence_length:], dtype=np.float32)

                z_score = self.predictor.predict(cp.asarray(recent_prices), cp.asarray(recent_volumes)) if self.use_gpu else compute_zscore_cpu(recent_prices, recent_volumes)

                model_input = np.stack([recent_prices, recent_volumes], axis=-1).reshape(1, -1, 2)
                multi_pred = self.model.predict(model_input, verbose=0)[0]
                model_score = np.mean(multi_pred) - recent_prices[-1]

                with self.lock:
                    combined_score = self.z_weight * z_score + self.model_weight * model_score

                if not self.filter_trade(z_score, model_score):
                    continue

                action = self.decision_engine.decide(combined_score, self.prices)
                if action in {"BUY", "SELL"}:
                    volatility = np.std(recent_prices)
                    size = max(1, int(volatility * 10))
                    self.execution_engine.execute(action, prices_chunk[-1], timestamps[-1], size=size)

                trade_result = prices_chunk[-1] - self.prices[-2] if len(self.prices) > 1 else 0
                self.recent_trades.append((z_score, model_score, trade_result))
                self.update_cache(z_score, model_score, trade_result)
                if len(self.recent_trades) > 1000:
                    self.recent_trades = self.recent_trades[-1000:]

            except Exception as e:
                print(f"Error in simulation: {e}")

        return self.execution_engine.final_report() if return_value else None