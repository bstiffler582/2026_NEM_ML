## Machine Learning in Automation

### Objectives
- Understand what ML is (and isn’t) in an industrial context
- Understand training and inference
    - Tools, technologies, roles & responsibilities
    - What is ONNX?
- TwinCAT real-time ML (TF38xx)

### Agenda
1. Context & Concepts
    - Use cases
        - Predictive maintenance
        - Anamoly detection
        - Quality classification
    - Supervised vs. unsupervised learning
    - Beckhoff products and solutions
2. Tooling & Setup
    - Python + packages
        - [Python 3.10](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
        - `pip install pandas numpy scikit-learn onnx skl2onnx matplotlib onnxruntime jupyter`
    - TwinCAT requirements
    - Sample data
3. Lab #1: Supervised learning
4. Lab #2: Unsupervised learning

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

For our process data, what meaningful features can we extract? Simply using the raw signal data is one approach, but that's not always effective or optimal for models - especially ones we want to run in real-time. Usually, better relationships can be derived by applying *aggregate functions* to the signal data. We don't have to get too fancy with it, either – a lot can be gained from your basic STATS 101 formulas:
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
We should be getting a fresh record every 1 second (cycle). We are gong to need a fair amount of records, so while this cooks 🧑‍🍳 let's talk about some concepts:

1. Supervised vs. unsupervised learning
    - We talked about **features**, now what are **labels**?


    
