import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model("HFT/model/instinct_model.keras")
scaler = joblib.load("HFT/model/scaler.pkl")

# dummy inputs
x = np.random.rand(10, 7).astype(np.float32)
x_scaled = scaler.transform(x)

print("→ Output predictions from model:")
print(model.predict(x_scaled))
