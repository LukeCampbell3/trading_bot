import matplotlib.pyplot as plt
import re
from datetime import datetime

with open("simulation.log", 'r') as f:
    raw_log_lines = f.readlines()

timestamps, actions, cash_vals, pos_vals, prices = [], [], [], [], []

for line in raw_log_lines:
    if "Executed" in line:
        try:
            ts_str = line.split(" - ")[0]
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
            match = re.search(r"Executed (.+?) at ([0-9.]+), Cash: ([0-9.]+), Pos: (\d+)", line)
            if match:
                action = match.group(1)
                price = float(match.group(2))
                cash = float(match.group(3))
                position = int(match.group(4))
                timestamps.append(ts)
                actions.append(action)
                prices.append(price)
                cash_vals.append(cash)
                pos_vals.append(position)
        except Exception:
            continue

portfolio_vals = [c + (p * pr) for c, p, pr in zip(cash_vals, pos_vals, prices)]

plt.figure(figsize=(14, 6))
plt.plot(timestamps, portfolio_vals, label='Portfolio Value', color='black')

for label, color, mark in [('BUY', 'green', '^'), ('SELL', 'red', 'v'), ('TRIGGER_SELL', 'orange', 'x'), ('TIME_EXIT', 'blue', 'x')]:
    idxs = [i for i, a in enumerate(actions) if label in a]
    if idxs:
        plt.scatter([timestamps[i] for i in idxs], [portfolio_vals[i] for i in idxs], 
                    color=color, label=label.replace('_', ' ').title(), marker=mark, s=40)

plt.title("Trade Log Visualization")
plt.xlabel("Time")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
