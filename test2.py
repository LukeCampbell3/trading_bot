import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
API_KEY = ""
TICKER = "TOP"
START_DATE = "2025-05-28"
END_DATE = "2025-06-07"
ATR_PERIOD = 14
VWAP_LOOKBACK = 20
CAPITAL_PER_TRADE = 1000  # USD
RISK_PER_TRADE = 0.01     # 1% risk per trade

# --- Polygon Data Fetching ---
def fetch_polygon_data(ticker, start_date, end_date, bar_limit=4000):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit={bar_limit}&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "results" not in data:
        raise Exception(f"Polygon API returned invalid response: {data}")

    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df["price"] = df["c"]
    df["volume"] = df["v"]
    df["high"] = df["h"]
    df["low"] = df["l"]
    df["close"] = df["c"]
    return df[["timestamp", "price", "volume", "high", "low", "close"]].copy()

# --- Indicator Computation ---
def compute_indicators(df):
    df.set_index("timestamp", inplace=True)
    df["VWAP"] = (df["price"] * df["volume"]).rolling(VWAP_LOOKBACK).sum() / df["volume"].rolling(VWAP_LOOKBACK).sum()

    df["TR"] = np.maximum(df["high"] - df["low"],
                          np.maximum(abs(df["high"] - df["close"].shift(1)),
                                     abs(df["low"] - df["close"].shift(1))))
    df["ATR"] = df["TR"].rolling(ATR_PERIOD).mean()
    df["RSI"] = compute_rsi(df["price"])
    df["Price_Change"] = df["price"].diff()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- Signal & Strategy Execution ---
def run_strategy(df):
    portfolio = []
    signals = []
    in_trade = False
    entry_price = 0
    shares = 0
    confidence = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        timestamp = df.index[i]

        # Skip if indicators not ready
        if pd.isna(row["VWAP"]) or pd.isna(row["ATR"]) or pd.isna(row["RSI"]):
            continue

        atr_median = df["ATR"].rolling(ATR_PERIOD).median().iloc[i]
        regime = None

        # --- Regime detection ---
        if row["ATR"] > atr_median and row["price"] > row["VWAP"]:
            regime = "Momentum"
        elif row["ATR"] < atr_median and abs(row["price"] - row["VWAP"]) < row["ATR"]:
            regime = "MeanReversion"

        if not in_trade and regime:
            signal_strength = 0

            # --- Entry Conditions ---
            if regime == "Momentum" and row["price"] > row["VWAP"] and row["volume"] > prev["volume"]:
                signal_strength = (row["price"] - row["VWAP"]) / max(row["ATR"], 1e-6)
                entry_price = row["price"]
                stop_price = entry_price - row["ATR"]
            elif regime == "MeanReversion" and row["RSI"] < 30 and row["price"] < row["VWAP"]:
                signal_strength = (row["VWAP"] - row["price"]) / max(row["ATR"], 1e-6)
                entry_price = row["price"]
                stop_price = entry_price - row["ATR"]

            if signal_strength > 0:
                risk_per_share = entry_price - stop_price
                size = int((CAPITAL_PER_TRADE * RISK_PER_TRADE) / max(risk_per_share, 1e-6))
                shares = size
                confidence = signal_strength
                in_trade = True
                signals.append(("BUY", timestamp, entry_price, shares))
                portfolio.append({"timestamp": timestamp, "value": 0})  # will update on next

        elif in_trade:
            current_price = row["price"]
            unrealized = (current_price - entry_price) * shares
            confidence_decay = (row["RSI"] > 70 or row["price"] < row["VWAP"])  # signal weakening
            confidence -= 0.05 if confidence_decay else -0.02  # smooth increase on strong signal
            confidence = max(0, min(confidence, 5))

            portfolio.append({"timestamp": timestamp, "value": unrealized})

            if confidence < 0.1:
                signals.append(("SELL", timestamp, current_price, shares))
                in_trade = False
                entry_price = 0
                shares = 0
                confidence = 0.0

    portfolio_df = pd.DataFrame(portfolio).set_index("timestamp")
    return signals, portfolio_df

# --- Visualization ---
def plot_results(df, signals, portfolio_df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["price"], label="Price", alpha=0.5)

    for sig in signals:
        label = "Buy" if sig[0] == "BUY" else "Sell"
        color = "green" if sig[0] == "BUY" else "red"
        plt.scatter(sig[1], sig[2], label=label, color=color, marker="^" if sig[0] == "BUY" else "v")

    plt.title(f"{TICKER} Trade Signals")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(portfolio_df.index, np.cumsum(portfolio_df["value"].fillna(0)), color="purple")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative P&L")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    df = fetch_polygon_data(TICKER, START_DATE, END_DATE)
    df = compute_indicators(df)
    signals, portfolio_df = run_strategy(df)

    for sig in signals:
        action, ts, price, size = sig
        print(f"[{ts}] {action}: {size} shares at ${price:.2f}")

    plot_results(df, signals, portfolio_df)
