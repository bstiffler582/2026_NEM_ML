import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load data
df = pd.read_csv("samples_scored.csv")

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

X = df[features]
y = df["man_quality"]

# Train
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=3)
model.fit(X, y)

# Export ONNX
initial_type = [('float_input', FloatTensorType([None, len(features)]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model exported.")