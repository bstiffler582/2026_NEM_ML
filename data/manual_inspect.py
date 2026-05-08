import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("samples.csv")

# delta idx - custom scoring feature
df["vibe_delta"] = (1 - abs(df["vibe_delta_idx"] - 12) / 12) ** 2
df["curr_delta"] = (1 - abs(df["curr_delta_idx"] - 12) / 12) ** 2

# quality function - operators "eyes"
df["man_quality"] = (
    1.2
        - 0.45 * df["vibe_delta"] 
        - 0.35 * df["curr_delta"] 
        - 0.10 * abs(df["vibe_delta_max"])
        - 0.01 * abs(df["curr_delta_max"]) 
        - 0.01 * df["curr_std"] 
        - 0.02 * df["vibe_std"]
)

# clamped to [0,1]
df["man_quality"] = df["man_quality"].clip(0, 1)

# visualize scoring distribution
print("\nQuality Summary:")
print(df["man_quality"].describe())

print("\nDistribution buckets:")
print(pd.cut(df["man_quality"], 
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]).value_counts())

# plot score distribution histogram
plt.hist(df["man_quality"], bins=20)
plt.title("Quality Distribution")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# save labeled dataset
df.drop(columns=["vibe_delta","curr_delta"], inplace=True)
df.to_csv("samples_manual_score.csv", index=False)