import pandas as pd
import cupy as cp
import numpy as np
import numba
import logging

logging.basicConfig(filename="simulation.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataHandler:
    def __init__(self, csv_path, chunk_size=100):
        self.data = pd.read_csv(csv_path, names=["timestamp", "price", "volume", "exchange", "conditions"])
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data.sort_values('timestamp', inplace=True)
        self.chunk_size = chunk_size
        self.index = 0
        self.total_rows = len(self.data)

    def stream_data(self):
        while self.index < self.total_rows:
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
    def __init__(self, base_threshold=1.5, min_threshold=0.7):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold

    def decide(self, score: float, prices: list):
        volatility = np.std(prices[-110:])
        dynamic_threshold = max(self.base_threshold * volatility, self.min_threshold)
        if score > dynamic_threshold:
            return "BUY"
        elif score < -dynamic_threshold:
            return "SELL"
        return "SIT"

class ExecutionEngine:
    def __init__(self, initial_capital=10000, cooldown_period=1):
        self.cash = initial_capital
        self.position = 0
        self.last_price = None
        self.trade_count = 0
        self.timestamps_seen = set()
        self.timestamps_missed = 0
        self.cooldown_period = cooldown_period
        self.cooldown_timer = 0

    def execute(self, action, price, timestamp):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        if timestamp in self.timestamps_seen:
            self.timestamps_missed += 1
            return
        self.timestamps_seen.add(timestamp)

        if action == "BUY" and self.cash >= price:
            self.position += 1
            self.cash -= price
            self.trade_count += 1
            self.cooldown_timer = self.cooldown_period

        elif action == "SELL" and self.position > 0:
            self.position -= 1
            self.cash += price
            self.trade_count += 1
            self.cooldown_timer = self.cooldown_period

        self.last_price = price
        logging.info(f"Executed {action} at {price}, Cash: {self.cash}, Pos: {self.position}")

    def final_report(self):
        portfolio_value = self.cash + (self.position * self.last_price if self.last_price else 0)
        print("\n--- Simulation Summary ---")
        print(f"Final Cash: ${self.cash:.2f}")
        print(f"Final Position: {self.position} shares")
        print(f"Final Price: ${self.last_price:.2f}" if self.last_price else "No trades executed.")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Total Trades Executed: {self.trade_count}")
        print(f"Timestamps Missed (duplicates): {self.timestamps_missed}")

class Simulation:
    def __init__(self, csv_path, use_gpu=True):
        self.data_handler = DataHandler(csv_path)
        self.use_gpu = use_gpu
        self.predictor = GPUPredictor()
        self.decision_engine = DecisionEngine()
        self.execution_engine = ExecutionEngine()
        self.prices = []

    def run(self):
        for chunk in self.data_handler.stream_data():
            try:
                prices_chunk = chunk['price'].values.astype(np.float32)
                volumes_chunk = chunk['volume'].values.astype(np.float32)
                timestamps = chunk['timestamp'].values

                self.prices.extend(prices_chunk.tolist())

                if len(self.prices) < 30:
                    continue

                recent_prices = np.array(self.prices[-30:], dtype=np.float32)
                recent_volumes = np.array(volumes_chunk[-30:], dtype=np.float32)

                if self.use_gpu:
                    score = self.predictor.predict(cp.asarray(recent_prices), cp.asarray(recent_volumes))
                else:
                    score = compute_zscore_cpu(recent_prices, recent_volumes)

                action = self.decision_engine.decide(score, self.prices)
                self.execution_engine.execute(action, prices_chunk[-1], timestamps[-1])
            except Exception as e:
                logging.error(f"Error in simulation loop: {e}")

        self.execution_engine.final_report()

def main():
    sim = Simulation(csv_path='HFT/archive/AAPL_2021_11.txt', use_gpu=True)
    sim.run()

main()
