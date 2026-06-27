import pandas as pd

df = pd.read_csv("trade_log.csv")
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')

exit_rows = df[df['action'].str.startswith("EXIT")]
valid_exits = exit_rows[~exit_rows['pnl'].isna()]

print("Total EXITs:", len(exit_rows))
print("Valid EXITs with PnL:", len(valid_exits))
