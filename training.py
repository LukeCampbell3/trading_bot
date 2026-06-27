# transformer_train.py
"""
Trainer that saves artifacts in a format alpaca_trader.py can load robustly.

Outputs (HFT/model):
- instinct_model.weights.h5    (PRIMARY: always loadable)
- instinct_model.keras         (optional convenience)
- scaler.pkl                   (RobustScaler fit on train only)
- model_manifest.json          (SEQ_LEN, FEATURES, arch params, horizon)

Key design choice:
- The trader should rebuild the model from manifest and call load_weights().
  This avoids Keras 3 weight-store mapping issues and custom object deserialization pitfalls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from tensorflow.keras import layers, Model

# ─── Configuration ───────────────────────────────────────────────
DATA_PATH = Path("archive")
MODEL_DIR = Path("HFT/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Feature computation params (must match live)
LOOKBACK = 100
FEATURES = ["z", "dz", "avg_dz", "ma_slope", "vol_surge", "dev_score", "trend"]
DZ_AVG_WINDOW = 5
VOL_WINDOW = 10
TREND_LEN = 20

# Sequence model params
SEQ_LEN = 60
HORIZON = 5
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10

# Transformer params (must match what trader rebuilds)
D_MODEL = 64
N_HEADS = 4
FF_DIM = 128
N_BLOCKS = 3
DROPOUT = 0.15

BATCH_SIZE = 256
EPOCHS = 25
LR = 2e-4


# ─── Data Loader ─────────────────────────────────────────────────
def load_batch_data(data_path: Path) -> pd.DataFrame:
    files = sorted(data_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_path.resolve()}")

    frames = []
    for f in files:
        df = pd.read_csv(f)

        if "Close" in df.columns:
            df.rename(columns={"Close": "price"}, inplace=True)
        if "Volume" in df.columns:
            df.rename(columns={"Volume": "volume"}, inplace=True)

        if "Date" in df.columns and "Time" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors="coerce",
            )
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if "price" not in df.columns or "volume" not in df.columns:
            raise ValueError(f"{f.name} missing required columns. Need price/volume or Close/Volume.")

        frames.append(df[["timestamp", "price", "volume"]].copy())

    out = pd.concat(frames, ignore_index=True)
    out.dropna(subset=["timestamp", "price", "volume"], inplace=True)
    out.sort_values("timestamp", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ─── Feature Engineering (same logic as live) ─────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["vwap"] = (df["price"] * df["volume"]).rolling(window=LOOKBACK).sum() / \
                 df["volume"].rolling(window=LOOKBACK).sum()
    df["std"] = df["price"].rolling(window=LOOKBACK).std()

    df["z"] = (df["price"] - df["vwap"]) / df["std"]
    df["prev_z"] = df["z"].shift(1)
    df["dz"] = df["z"] - df["prev_z"]
    df["avg_dz"] = df["dz"].rolling(window=DZ_AVG_WINDOW).mean()

    df["ma_recent"] = df["price"].rolling(20).mean()
    df["ma_prev"] = df["price"].shift(10).rolling(20).mean()
    df["ma_slope"] = df["ma_recent"] - df["ma_prev"]

    vol_mean = df["volume"].rolling(VOL_WINDOW).mean()
    df["vol_surge"] = (df["volume"] - vol_mean) / vol_mean

    df["dev_score"] = (df["vwap"] - df["price"]) / df["std"]

    df["trend"] = df["price"].rolling(window=TREND_LEN).apply(
        lambda x: np.polyfit(range(TREND_LEN), x, 1)[0] if len(x) == TREND_LEN else np.nan,
        raw=True,
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── Sequence Builder ────────────────────────────────────────────
def make_sequences(df: pd.DataFrame, feature_cols, seq_len: int, horizon: int):
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    price = df["price"].to_numpy(dtype=np.float32)

    fwd = (price[horizon:] - price[:-horizon]) / price[:-horizon]
    X_list, y_list = [], []

    max_t_end = len(fwd) - 1
    for t_end in range(seq_len - 1, max_t_end + 1):
        t0 = t_end - (seq_len - 1)
        X_list.append(feats[t0:t_end + 1])
        y_list.append(fwd[t_end])

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    y = np.clip(y, -0.1, 0.1)
    return X, y


# ─── Transformer Model ───────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)

        key_dim = max(1, self.d_model // max(1, self.num_heads))
        self.attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.d_model),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(self.dropout)
        self.drop2 = layers.Dropout(self.dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.ln1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.ln2(x + self.drop2(ffn_out, training=training))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return cfg

def build_model(seq_len: int, n_features: int,
                d_model: int = D_MODEL,
                n_heads: int = N_HEADS,
                ff_dim: int = FF_DIM,
                n_blocks: int = N_BLOCKS,
                dropout: float = DROPOUT) -> Model:
    inp = layers.Input(shape=(seq_len, n_features), name="x")

    # learned positional embedding
    @tf.keras.utils.register_keras_serializable(package="Custom")
    class PositionalEmbedding(layers.Layer):
        def __init__(self, seq_len: int, d_model: int, **kwargs):
            super().__init__(**kwargs)
            self.seq_len = int(seq_len)
            self.d_model = int(d_model)
            self.pos_emb = layers.Embedding(input_dim=self.seq_len, output_dim=self.d_model)

        def call(self, x):
            # x: (batch, T, d_model)
            T = tf.shape(x)[1]
            positions = tf.range(start=0, limit=T, delta=1)        # (T,)
            pos = self.pos_emb(positions)                          # (T, d_model)
            return x + pos                                         # broadcast over batch

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
            return cfg

    # project features -> d_model
    x = layers.Dense(D_MODEL, name="proj")(inp)
    x = PositionalEmbedding(seq_len, D_MODEL, name="pos")(x)
    x = layers.Dropout(DROPOUT, name="drop_in")(x)

    for i in range(n_blocks):
        x = TransformerBlock(d_model=d_model, num_heads=n_heads, ff_dim=ff_dim, dropout=dropout, name=f"tb{i}")(x)

    # pool (last token + global average)
    last = layers.Lambda(lambda t: t[:, -1, :], name="last_tok")(x)
    avg = layers.GlobalAveragePooling1D(name="avg_pool")(x)
    x = layers.Concatenate(name="pool_cat")([last, avg])

    # head
    x = layers.Dense(128, activation="gelu", name="head_128")(x)
    x = layers.Dropout(dropout, name="drop_128")(x)
    x = layers.Dense(64, activation="gelu", name="head_64")(x)
    x = layers.Dropout(dropout, name="drop_64")(x)

    out = layers.Dense(1, activation="linear", name="y")(x)

    model = Model(inp, out, name="instinct_transformer")
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="mse")
    return model


# ─── Train ───────────────────────────────────────────────────────
def main():
    raw = load_batch_data(DATA_PATH)
    raw = compute_features(raw)

    X, y = make_sequences(raw, FEATURES, seq_len=SEQ_LEN, horizon=HORIZON)

    n = len(X)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * (TRAIN_FRAC + VAL_FRAC))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val     = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test   = X[n_val:], y[n_val:]

    # Scale features: fit scaler on train only, apply to all
    scaler = RobustScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    def scale_3d(X3):
        X2 = X3.reshape(-1, X3.shape[-1])
        X2s = scaler.transform(X2)
        return X2s.reshape(X3.shape)

    X_train_s = scale_3d(X_train)
    X_val_s   = scale_3d(X_val)
    X_test_s  = scale_3d(X_test)

    # Save scaler
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    model = build_model(seq_len=SEQ_LEN, n_features=len(FEATURES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    )

    test_loss = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"\nTest MSE: {test_loss:.8f}")

    # ── SAVE IN TRADER-FRIENDLY FORMAT ───────────────────────────
    weights_path = MODEL_DIR / "instinct_model.weights.h5"
    keras_path   = MODEL_DIR / "instinct_model.keras"
    manifest_path = MODEL_DIR / "model_manifest.json"

    # IMPORTANT: run one forward pass so all variables are created
    _ = model(tf.zeros((1, SEQ_LEN, len(FEATURES)), dtype=tf.float32), training=False)

    # Save weights FIRST (this is what the trader should load)
    model.save_weights(weights_path)

    # Save full model (optional convenience)
    model.save(keras_path)

    # Save manifest for trader rebuild
    manifest = {
        "SEQ_LEN": SEQ_LEN,
        "HORIZON": HORIZON,
        "FEATURES": FEATURES,
        "LOOKBACK": LOOKBACK,
        "ARCH": {
            "D_MODEL": D_MODEL,
            "N_HEADS": N_HEADS,
            "FF_DIM": FF_DIM,
            "N_BLOCKS": N_BLOCKS,
            "DROPOUT": DROPOUT,
        }
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n Training complete. Artifacts saved:")
    print(" -", weights_path)
    print(" -", keras_path)
    print(" -", manifest_path)
    print(" -", MODEL_DIR / "scaler.pkl")

    # ── SELF-VERIFY LOAD PATH (rebuild + load_weights) ───────────
    m2 = build_model(seq_len=SEQ_LEN, n_features=len(FEATURES))
    _ = m2(tf.zeros((1, SEQ_LEN, len(FEATURES)), dtype=tf.float32), training=False)
    m2.load_weights(weights_path)

    # sanity inference
    _ = m2.predict(X_val_s[:2], verbose=0)
    print(" Verified: rebuild + load_weights works.")


if __name__ == "__main__":
    main()