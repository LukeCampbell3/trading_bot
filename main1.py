import pandas as pd
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

class ImprovedPLStrategy:
    def __init__(self, initial_capital=10000, max_hold_ticks=40):
        self.capital = initial_capital
        self.position = 0
        self.last_price = None
        self.last_buy_price = None
        self.max_unrealized = 0
        self.cooldown_timer = 0
        self.time_in_trade = 0
        self.max_hold_ticks = max_hold_ticks
        self.prev_profit = 0
        self.trade_log = []

    def run(self, df):
        np_prices = df["price"].values.astype(np.float32)
        np_volumes = df["volume"].values.astype(np.float32)
        timestamps = df["timestamp"].values

        dz_history = []
        results = []

        for i in range(100, len(np_prices)):
            # Move only the windows to GPU
            price_window = cp.asarray(np_prices[i - 100:i])
            volume_window = cp.asarray(np_volumes[i - 100:i])
            price = float(np_prices[i])  # CPU-side
            timestamp = timestamps[i]

            # GPU-calculated stats
            vwap = cp.sum(price_window * volume_window) / cp.sum(volume_window)
            std = cp.std(price_window)
            z = ((price_window[-1] - vwap) / std).item() if std > 0 else 0.0
            prev_z = ((price_window[-2] - vwap) / std).item() if std > 0 else 0.0
            dz = z - prev_z

            dz_history.append(dz)
            if len(dz_history) > 10:
                dz_history.pop(0)
            avg_dz = float(np.mean(dz_history))  # CPU

            # Slope via GPU
            short_ma = cp.mean(price_window[-20:])
            prev_ma = cp.mean(price_window[-30:-10]) if i >= 30 else short_ma
            ma_slope = (short_ma - prev_ma).item()

            volume_now = float(np_volumes[i])
            volume_mean = cp.mean(cp.asarray(np_volumes[i - 10:i])).item()
            vol_surge = (volume_now - volume_mean) / volume_mean if volume_mean > 0 else 0.0

            deviation_score = ((vwap - price_window[-1]) / std).item() if std > 0 else 0.0
            instinct_score = deviation_score + 1.2 * avg_dz + 0.5 * ma_slope + 0.6 * vol_surge

            action = "SIT"
            realized_pnl = None

            if self.cooldown_timer > 0:
                self.cooldown_timer -= 1

            threshold = 0.8
            if self.position == 0 and instinct_score > threshold and self.cooldown_timer == 0:
                position_scale = 0.03 if self.prev_profit <= 0 else 0.045
                position_size = max(1, int((self.capital * position_scale * min(abs(instinct_score), 3)) // price))
                if self.capital >= price * position_size:
                    self.capital -= price * position_size
                    self.position += position_size
                    self.last_buy_price = price
                    self.time_in_trade = 0
                    self.max_unrealized = 0
                    self.cooldown_timer = 5
                    action = f"BUY({position_size})"
                    self.log_trade(timestamp, action, price, instinct_score, z, dz, avg_dz, ma_slope, vol_surge, deviation_score)

            if self.position > 0:
                self.time_in_trade += 1
                unrealized = price - self.last_buy_price
                self.max_unrealized = max(self.max_unrealized, unrealized)
                profit_ratio = unrealized / self.last_buy_price

                exit_signal = False
                if profit_ratio > 0.004 and unrealized < self.max_unrealized - 0.006 * self.last_buy_price:
                    exit_signal = True
                if profit_ratio < -0.02 or self.time_in_trade > self.max_hold_ticks:
                    exit_signal = True
                if avg_dz < -0.002 and profit_ratio > 0.001:
                    exit_signal = True

                if exit_signal:
                    self.capital += price * self.position
                    realized_pnl = (price - self.last_buy_price) * self.position
                    self.prev_profit = profit_ratio
                    self.position = 0
                    self.last_buy_price = None
                    self.cooldown_timer = 10 if profit_ratio < 0 else 3
                    self.time_in_trade = 0
                    self.max_unrealized = 0
                    action = "EXIT"
                    self.log_trade(timestamp, action, price, instinct_score, z, dz, avg_dz, ma_slope, vol_surge, deviation_score, realized_pnl)

            portfolio_value = self.capital + self.position * price
            self.last_price = price
            results.append((timestamp, action, portfolio_value, price))

        self.export_log()
        return pd.DataFrame(results, columns=["timestamp", "action", "portfolio_value", "price"])

    def log_trade(self, timestamp, action, price, instinct_score, z, dz, avg_dz, ma_slope, vol_surge, deviation_score, realized_pnl=None):
        self.trade_log.append({
            "timestamp": timestamp,
            "action": action,
            "price": price,
            "instinct_score": instinct_score,
            "z": z,
            "dz": dz,
            "avg_dz": avg_dz,
            "ma_slope": ma_slope,
            "vol_surge": vol_surge,
            "deviation_score": deviation_score,
            "pnl": realized_pnl
        })

    def export_log(self):
        pd.DataFrame(self.trade_log).to_csv("trade_log.csv", index=False)

# --- Simulation Runner ---
def simulate_strategy(csv_path):
    df = pd.read_csv(csv_path, names=["timestamp", "price", "volume", "exchange", "conditions"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    strategy = ImprovedPLStrategy()
    results = strategy.run(df)

    plt.figure(figsize=(14, 6))
    plt.plot(results["timestamp"], results["portfolio_value"], color="teal", label="Portfolio Value")
    plt.title("Valuation-Driven Adaptive P&L Strategy (Hybrid CPU-GPU)")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    final = results.iloc[-1]
    print("\n--- Simulation Summary ---")
    print(f"Final Portfolio Value: ${final['portfolio_value']:.2f}")
    print(f"Final Price: ${final['price']:.2f}")
    print(f"Total Trades Executed: {len(results[results['action'] != 'SIT'])}")
    print(f"Data Points Processed: {len(results)}")

if __name__ == "__main__":
    simulate_strategy("HFT/archive/AAPL_2021_11.txt")