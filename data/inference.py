import pandas as pd
import numpy as np
import onnxruntime as ort

# =========================
# Load ONNX
# =========================
sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name

# =========================
# Load test data
# =========================
df = pd.read_csv("samples.csv")

features = [
    "temp",
    "cycle_time",
    "curr_mean",
    "curr_peak",
    "curr_range",
    "curr_delta_max",
    "curr_delta_idx",
    "curr_std",
    "vibe_mean",
    "vibe_peak",
    "vibe_range",
    "vibe_delta_max",
    "vibe_delta_idx",
    "vibe_std"
]

X = df[features].values.astype(np.float32)

# =========================
# Run inference
# =========================
preds = sess.run(None, {input_name: X})[0]

df["onnx_pred"] = preds

df.to_csv("inferred.csv")