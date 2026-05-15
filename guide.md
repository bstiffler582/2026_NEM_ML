## Machine Learning in Automation

### Objectives
- Understand ML in an industrial context
    - What can it do?
    - Where does it make sense?
- Understand training and inference
    - Tools, technologies, roles & responsibilities

### Agenda
1. Context & Concepts (Presentation)
2. Tooling & Setup
    - Python + packages
        - [Python 3.10](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
        - `pip install pandas numpy scikit-learn onnx skl2onnx matplotlib onnxruntime`
        - Set `PATH` environment variable
    - TwinCAT requirements
        - TF3800 Machine Learning Runtime
3. Lab: Supervised learning

## Lab #1 - Supervised Learning

#### Visionless Quality Inspection
Customer problem statement:
>🧑‍🏭 "Currently, parts are inspected at regular intervals to ensure quality. This manual step is slow and inconsistent. We know there is a relationship between the in-process sensor data and the quality of a part, but we have not been able to substantiate it.
>
>The crucial part of the process takes roughly 250ms, and contains the following relevant signals:
>- Motor current
>- Vibration
>- Temperature
>- Cycle time
>
>Upon viewing and trending these signals, some relationships are obvious; e.g. higher temperatures cause volatility in the process. Similarly, increases in vibration and current *correlate* to lower quality parts, but do not reliably predict quality with any consistency. **We have tried implementing conditional logic, but it feels like we are chasing our tails!**"

#### Feature Extraction

The above metrics represent our data set, but they are not the only features we can extract.

> Example: **what's in a timestamp?**<br />
> A timestamp might look like this: *1778118721000* or this: *YYYY:MM:DD MM:HH:ss*<br />
> But what else does it tell us? DoW, DoM, ToD, age/recency vs NOW(), interval patterns like daily, weekly, monthly, per-shift, seasonal, etc... → features can be **derived** from the context of a value.

For our process data, what meaningful features can we extract? Simply using the raw signal data is one approach, but that's not always effective or optimal for models - especially ones we want to run in real-time. Usually, better relationships can be derived by applying *aggregate functions* or signal processing to the raw data. We don't have to go full condition monitoring with it, either – a lot can be gained from your basic STATS 101 formulas:
- Mean (the average)
- Range (the spread)
- Standard deviation (stability)
- Delta & Delta Idx (significant rates of change)
- Peaks/valleys (anomalies)

These derived values will give our machine learning model more data and more context on which to train and infer.

> **Note:** A typical workflow might have us capturing this raw signal data *first*, then apply these functions as a *post-processing* step; via database queries, from R or python scripts, etc...<br />
>We are going to do it right in the PLC. Why? Because:<br />
>a.) We will be inferencing right in the real-time<br />
>b.) We will need this *processed* data for the inference input<br />
>c.) We can :)

#### Data Acquisition
1. Create a structure to hold all the model's features:
```js
TYPE ST_Features :
STRUCT
	temp				: LREAL;
	cycle_time			: DINT;
	
	curr_mean			: LREAL;
	curr_peak			: LREAL;
	curr_range			: LREAL;
	curr_delta_max		: LREAL;
	curr_delta_idx		: LREAL;
	curr_std			: LREAL;
	                	
	vibe_mean			: LREAL;
	vibe_peak			: LREAL;
	vibe_range			: LREAL;
	vibe_delta_max		: LREAL;
	vibe_delta_idx		: LREAL;
	vibe_std			: LREAL;
END_STRUCT
END_TYPE
```
##### *Question: why no derived labels for temperature or cycle time?

2. To populate the feature structure, we will use this handy `FB_ALY_ArrayStatistics` block from the `Tc3_Analytics` library. It packages up all the aggregate functions we need:
```js
    // declare user vars
	fbStatsCurrent		: FB_ALY_ArrayStatistics;
	fbStatsVibration	: FB_ALY_ArrayStatistics;
	stFeatures			: ST_Features;
```
```js
    /// init ALY FBs
	fbStatsCurrent.Configure(FALSE, 0.1, 0.1, 0);
	fbStatsVibration.Configure(FALSE, 0.1, 0.1, 0);
```
```js
    /// populate non-aggregate features
	stFeatures.temp := fTemperature;
	stFeatures.cycle_time := nCycleTime;
```
```js
    /// call ALY FBs
	fbStatsCurrent.Call(ADR(current_arr), SIZEOF(current_arr));
	fbStatsVibration.Call(ADR(vib_arr), SIZEOF(vib_arr));
	
	/// populate aggregate features
	IF fbStatsCurrent.bNewResult THEN
		stFeatures.curr_mean := fbStatsCurrent.fMean;
		stFeatures.curr_peak := fbStatsCurrent.fMax;
		stFeatures.curr_range := fbStatsCurrent.fMax - fbStatsCurrent.fMin;
		stFeatures.curr_delta_max := fbStatsCurrent.fMaxDelta;
		stFeatures.curr_delta_idx := fbStatsCurrent.nIdxMaxDelta;
		stFeatures.curr_std := fbStatsCurrent.fStandardDeviation;
	END_IF
	IF fbStatsVibration.bNewResult THEN
		stFeatures.vibe_mean := fbStatsVibration.fMean;
		stFeatures.vibe_peak := fbStatsVibration.fMax;
		stFeatures.vibe_range := fbStatsVibration.fMax - fbStatsVibration.fMin;
		stFeatures.vibe_delta_max := fbStatsVibration.fMaxDelta;
		stFeatures.vibe_delta_idx := fbStatsVibration.nIdxMaxDelta;
		stFeatures.vibe_std := fbStatsVibration.fStandardDeviation;
	END_IF
	
	/// generate CSV string
	IF fbStatsCurrent.bNewResult AND fbStatsVibration.bNewResult THEN
		sData := F_Feature_Csv(stFeatures);
	END_IF
```
3. Let's collect some data. Ensure you have a valid path in the `sFilePath` variable. Copy the [headers.csv file](/data/headers.csv) into that path and rename it appropriately ("samples.csv"). Flip the `bEnableLogging` bit and verify that data is coming in:
```
temp,cycle_time,curr_mean,curr_peak,curr_range,curr_delta_max,curr_delta_idx,curr_std,vibe_mean,vibe_peak,vibe_range,vibe_delta_max,vibe_delta_idx,vibe_std
33.7999,228,8.4130,9.7077,4.8909,-1.2612,23.0,1.3170,1.9017,2.7450,1.7530,-1.4612,22.0,0.4821
33.7387,224,8.1583,9.5316,4.1955,1.0633,5.0,1.1701,1.9758,2.7215,1.4106,-0.9529,12.0,0.3460
...
```
We should be getting a fresh record every 1 second (cycle). In ML, usually more data = better, so let's let this cook for awhile 🧑‍🍳 while we talk about some concepts and terminology:

1. Supervised vs. unsupervised learning
    - We talked about **features**, now what are **labels**?
	- Supervised example: regression trees (this is what we will use for the lab)
2. Bias and variance - balancing model error
    - High Bias: underfit, oversimplified, too general
        - Need more features, increase complexity
        - "bias" - think "presumptuous"
    - High Variance: overfit, complex, noisy
        - Need more training data, reduce complexity
        - "variance" - between point-to-point relationships

Now we should have a reasonably-sized data set to work with. Step 1 will be **training**. Since we are using a *supervised* learning algorithm, first we need to apply **labels**. For the "manual" inspection, we will use the following python script to simulate an operator applying a score to each part in our data set.

```python
# manual_inspect.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("samples_train.csv")

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
```

Ideally you will be able to feed this script roughly 1000 samples. If you have more than that, trim the file and keep the excess for testing with the trained model. Make sure to keep all the appropriate headers intact when modifying CSVs.

Double-check your file names and paths, and run the script with `python .\manual_inspect.py`. You should get a histogram of the scoring distribution, and an output file `samples_manual_score.csv` with all the same data + a new column with the manual score between `0.0-1.0`. Let's perform some manual analysis of the data set in Excel. We can trend the score alongside some of the features and illustrate some of the correlations mentioned in the customer problem statement.

> Hint: The manual score most heavily correlates with vibration and current data. Not just the mean, peak, or max range, *but the point in the process at which each signal varies the most.* This is heavily influenced by temperatue and cycle time - but only when those conditions complement each other in certain ways. Think about the difficulty in capturing all of this effectively with conditional logic:
```js
// temp in threshold AND cycle time in threshold AND 
// difference between vibration AND current measurements are outside acceptable range...
if ((temp in temp_thresh_1) && !(temp in temp_thresh_2) && (cycles in cycle_thresh_1) || !(temp in temp_thresh_1) && (cycles in cycle_thresh_2))
	for (i in (cycles in cycle_thresh))
		if (abs(vib_data[i + 1] - vib_data[i]) > vib_delta_thresh) && 
			(abs(curr_data[i + 1] - curr_data[i]) > curr_delta_thresh)
				//...
```
...and this is still not guaranteed to produce results consistent with the "eyeball" test. This is what the customer meant by "chasing [their] tails" with regards to an algorithmic approach. These are examplary conditions for the *when to use it* question in regards to the application of machine learning.

So, let's start building our regression trees from scratch! JK, the python tools package that bit up for us into nice 1- or 2-line function calls. With the following script:

```python
```