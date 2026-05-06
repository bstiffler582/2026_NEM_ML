import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
df = pd.read_csv("samples.csv")

# =========================
# Quality function
# =========================
df["quality"] = (
    2.0
    - 0.10 * df["vibe_peak"]
    - 0.10 * (df["curr_range"] - 4.0)
    - 0.10 * df["vibe_range"]
    - 0.10 * np.abs(df["temp"] - 32.0) / 20.0
    - 0.10 * df["press_std"]
)

# Optional interaction (recommended)
#df.loc[
#    (df["curr_delta"] > 0.066) & (df["curr_range"] > 7.0),
#    "quality"
#] -= 0.3

#df.loc[
#    (df["temp"] > 41) & (df["curr_std"] > 1.9),
#    "quality"
#] -= 0.3

# Clamp to [0,1]
df["quality"] = df["quality"].clip(0, 1)

# =========================
# Quick stats
# =========================
print("\nQuality Summary:")
print(df["quality"].describe())

print("\nDistribution buckets:")
print(pd.cut(df["quality"], bins=[0, 0.5, 0.8, 1.0]).value_counts())

# =========================
# Plot histogram
# =========================
plt.hist(df["quality"], bins=20)
plt.title("Quality Distribution")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# =========================
# Save updated dataset
# =========================
df.to_csv("data_with_quality.csv", index=False)
# print("\nSaved to data_with_quality.csv")