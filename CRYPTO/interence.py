import pickle, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path

SEQ_LEN = 120
cache   = Path("cache")

model   = tf.keras.models.load_model(cache / "btc_transformer.keras",
                                     custom_objects={"PositionalEncoding": PositionalEncoding})
with open(cache / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_next(df_live: pd.DataFrame):
    # df_live must already have the same feature columns in the same order
    arr = scaler.transform(df_live)[-SEQ_LEN:]      # last 120 rows
    x   = arr[np.newaxis, ...]                      # (1, seq_len, feat_dim)
    price_scaled, dir_prob = model.predict(x, verbose=0)
    price = price_scaled[0,0] * scaler.scale_[df_live.columns.get_loc("close")] + \
            scaler.mean_[df_live.columns.get_loc("close")]
    return float(price), float(dir_prob[0,0])
