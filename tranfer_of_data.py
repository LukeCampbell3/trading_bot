import pandas as pd

df = pd.read_feather("HFT\\data_by_days\\train_0.feather")
df.to_csv("train_0.csv", index=False)
