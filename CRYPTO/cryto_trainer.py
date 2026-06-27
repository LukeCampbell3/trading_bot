# trainer.py  –  Optimized Bitcoin minute-level trend predictor

import os, pickle, random
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW

# ────────────────────────── CONFIG ──────────────────────────
CFG = dict(
    MINUTE_CSV   = "CRYPTO/data/btcusd_1min.csv",
    DAILY_CSV    = "CRYPTO/data/btcusd_1day.csv",
    CACHE_DIR    = "cache",
    SEQ_LEN      = 120,
    BATCH        = 512,
    EPOCHS       = 20,
    LR           = 3e-4,
    D_MODEL      = 64,
    NUM_HEADS    = 4,
    FF_DIM       = 128,
    DROPOUT      = 0.1,
    VAL_SPLIT    = 0.15,
    RANDOM_SEED  = 42,
)

os.makedirs(CFG["CACHE_DIR"], exist_ok=True)
tf.random.set_seed(CFG["RANDOM_SEED"])
np.random.seed(CFG["RANDOM_SEED"])
random.seed(CFG["RANDOM_SEED"])

# ──────────────────── DATA PREP ──────────────────────
def load_and_merge(minute_path: str, daily_path: str) -> pd.DataFrame:
    df_m = pd.read_csv(minute_path)
    df_m["timestamp"] = pd.to_datetime(df_m["Timestamp"], unit="s", utc=True)
    df_m = df_m.drop(columns=["Timestamp"]).rename(columns=str.lower).set_index("timestamp")

    df_d = pd.read_csv(daily_path, parse_dates=["Date"])
    df_d = df_d.drop(columns=["Adj Close"]).rename(columns=str.lower)
    df_d["date"] = pd.to_datetime(df_d["date"], utc=True)
    df_d = df_d.set_index("date").add_prefix("d_")

    df_m["date"] = df_m.index.floor("D")
    df = df_m.join(df_d, on="date").drop(columns=["date"])
    return df.ffill().dropna()

df = load_and_merge(CFG["MINUTE_CSV"], CFG["DAILY_CSV"]).iloc[-1_000_000:]  # Reduce to last 1M minutes
feature_cols = df.columns.tolist()
print(f"Loaded shape ⇒ {df.shape}, features: {feature_cols}")

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df).astype(np.float16), index=df.index, columns=df.columns)
# Reduce dataset for faster training (last 250k rows)
#df_scaled = df_scaled.iloc[-(250_000 + CFG["SEQ_LEN"]):]


with open(Path(CFG["CACHE_DIR"]) / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
(Path(CFG["CACHE_DIR"]) / "features.txt").write_text("\n".join(feature_cols))

def make_windows_fast(arr, seq_len):
    n = arr.shape[0] - seq_len
    X = np.lib.stride_tricks.sliding_window_view(arr, (seq_len, arr.shape[1]))[:-1, 0]
    y_price = arr[seq_len:, df.columns.get_loc("close")]
    prev_close = arr[seq_len - 1:-1, df.columns.get_loc("close")]
    y_dir = (y_price > prev_close).astype(np.float32)
    return X, y_price, y_dir

X, y_price, y_dir = make_windows_fast(df_scaled.values, CFG["SEQ_LEN"])
print("Windowed:", X.shape, y_price.shape, y_dir.shape)

split = int(len(X) * (1 - CFG["VAL_SPLIT"]))
X_tr, X_val = X[:split], X[split:]
yp_tr, yp_val = y_price[:split], y_price[split:]
yd_tr, yd_val = y_dir[:split], y_dir[split:]

# ─────────────────── LR SCHEDULE ──────────────────────
first_decay_steps = len(X_tr) // CFG["BATCH"] * 2  # ~2 epochs
lr_schedule = CosineDecayRestarts(
    initial_learning_rate=CFG["LR"],
    first_decay_steps=first_decay_steps,
    t_mul=2.0,
    m_mul=0.9,
    alpha=1e-6
)

# ───────────────── MODEL ─────────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, **kw):
        super().__init__(**kw)
        self.d_model = d_model

    def call(self, x):
        pos = tf.range(tf.shape(x)[-2], dtype=tf.float32)[..., tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, ...]
        angle = pos / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
        pos_encoding = tf.where(tf.cast(i, tf.int32) % 2 == 0, tf.sin(angle), tf.cos(angle))
        return x + tf.cast(pos_encoding, x.dtype)

def build_model(seq_len: int, feat_dim: int) -> models.Model:
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Dense(CFG["D_MODEL"])(inp)
    x = PositionalEncoding(CFG["D_MODEL"])(x)
    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=CFG["NUM_HEADS"], key_dim=CFG["D_MODEL"], dropout=CFG["DROPOUT"])(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        ffn = layers.Dense(CFG["FF_DIM"], activation="gelu")(x)
        ffn = layers.Dense(CFG["D_MODEL"])(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)

    x_flat = layers.GlobalAveragePooling1D()(x)
    price_out = layers.Dense(1, name="price")(x_flat)
    dir_out = layers.Dense(1, activation="sigmoid", name="direction")(x_flat)
    return models.Model(inp, outputs=[price_out, dir_out])

model = build_model(CFG["SEQ_LEN"], X.shape[-1])
losses = {"price": "mse", "direction": "binary_crossentropy"}
loss_weights = {"price": 1.0, "direction": 0.2}

# ─────────────────── CALLBACKS ───────────────────
cb = [
    callbacks.ReduceLROnPlateau(monitor="val_price_mae", patience=3, factor=0.5, verbose=1),
    callbacks.EarlyStopping(monitor="val_price_mae", patience=5, restore_best_weights=True)
]

# ──────────────── STAGE 1: Fast AdamW Training ────────────────
print("\nStage 1: Fast convergence with AdamW + CosineDecay\n")
opt_stage1 = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
model.compile(optimizer=opt_stage1, loss=losses, loss_weights=loss_weights,
              metrics={"price": "mae", "direction": "accuracy"})
model.fit(X_tr, {"price": yp_tr, "direction": yd_tr},
          validation_data=(X_val, {"price": yp_val, "direction": yd_val}),
          epochs=10, batch_size=CFG["BATCH"], callbacks=cb, verbose=1)

# ──────────────── STAGE 2: Fine-tune with Clipped AdamW ────────────────
print("\nStage 2: Fine-tuning with clipped AdamW + cosine restarts\n")
opt_stage2 = AdamW(learning_rate=lr_schedule, weight_decay=1e-5, clipnorm=1.0)
model.compile(optimizer=opt_stage2, loss=losses, loss_weights=loss_weights,
              metrics={"price": "mae", "direction": "accuracy"})
model.fit(X_tr, {"price": yp_tr, "direction": yd_tr},
          validation_data=(X_val, {"price": yp_val, "direction": yd_val}),
          initial_epoch=10, epochs=CFG["EPOCHS"], batch_size=CFG["BATCH"], callbacks=cb, verbose=1)

# ──────────────── Save Model ────────────────
model.summary()
model.save(Path(CFG["CACHE_DIR"]) / "btc_transformer_hybrid.keras")
print("\n✅ Training complete. Model & scaler saved to:", CFG["CACHE_DIR"])