## Machine Learning in Automation

### Objectives
- Understand what ML is (and isn’t) in an industrial context
- Know when to use supervised vs. unsupervised learning
- Training a simple model
    - Exporting it to ONNX
- Run inference on new data (via TF3800)

### Agenda
1. Context & Concepts
    - Real-time control versus real-time ML
    - Use cases
        - Predictive maintenance
        - Anamoly detection
        - Quality classification
    - Supervised vs. unsupervised learning (labels)
    - Beckhoff products and solutions
    - Competing products and solutions
2. Tooling & Setup
    - Python + packages
        - [Python 3.10](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
        - `pip install pandas numpy scikit-learn onnx skl2onnx matplotlib onnxruntime jupyter`
    - TwinCAT requirements
    - Sample data
3. Lab #1: Supervised learning
4. Lab #2: Unsupervised learning

## Lab #1 - Supervised Learning

"Visionless Quality Inspection"
>Currently, parts are manually inspected at regular intervals to ensure quality. This manual process is slow and inconsistent. We believe there is likely some relationship between the in-process sensor data and the quality of a part. Ideally we can leverage this to reduce the frequency of manual inspections.

The (important part of the) process takes roughly 200ms. We have implemented a high-speed data logger to capture the following points every PLC scan (10ms) within this time window across several machine cycles:
- Motor current
- Temperature
- Vibration
- Pressure
- Cycle time

We can view and trend the data, but it does not paint a clear picture of the relationships we are searching for. What should we do next?

### Feature Extracting

The above metrics represent our data set, but they are not the only features we can extract.

Feature extraction example: what's in a timestamp?
> YYYY:MM:DD MM:HH:ss; but also DoW, DoM, ToD, *age*/*recency* (context of NOW), etc.

> Captures patterns (daily, weekly, monthly, shift, seasonal)

So for our process data, what meaningful features can we extract? Raw time series data is usable, but often not effective or optimized for model input. Take the captured window of input data and apply STATS 101:
- Mean (meaures the average)
- Range (measures the spread)
- Standard deviation (measures the stability)
- Max Delta (rate of change)
- Peaks / valleys (captures significant events)

"non-obvious temporal features"

#### The plan:
1. Data acqusition
    - Capture several cycles worth of *raw* data
    - **Label** the results (via "manual" inspection)
    - ???
    - Profit
2. Apply aggregate functions
    - Establish relationships:
        - What effect does cycle time, temperature, pressure have on motor current?
        - On vibration?
        - How would we leverage them algorithmically?
        - Can Machine Learning save us time?
    - Create mapping between **features** and **labels**


    
