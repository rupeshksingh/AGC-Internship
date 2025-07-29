# Hybrid Modeling: STL Decomposition with XGBoost on Residuals (Week 7)

This document outlines the work from the seventh week, which explored a hybrid modeling strategy. This approach involved decomposing the time series using STL and then training an XGBoost model specifically on the residual component to detect anomalies.

---

## 🎯 Objective

The goal for this week was to test a "divide and conquer" strategy for anomaly detection. The main objective was to determine if decomposing the time series and modeling its components separately could yield better results. Specifically, the hypothesis was that anomalies, being irregular events, would be most prominent in the **residual** component of the signal after the trend and seasonal components were removed.

---

## ⚙️ Methodology

The experiment, detailed in the `STL_XGB.py` script, followed a two-step process:

### **1. STL Decomposition**
First, the **Seasonal-Trend-Loess (STL)** algorithm was applied to the roller speed time series. STL is a powerful and versatile method that breaks down a time series into three components:
* **Trend**: The long-term progression of the series.
* **Seasonal**: The periodic patterns in the data.
* **Residual**: The remaining, irregular part of the series after the trend and seasonality have been removed.

The primary goal of this step was to isolate the residual component, where it was believed the sudden fluctuations from glass breakage would be most clearly represented.

### **2. XGBoost on Residuals**
Once the residual component was extracted, an **XGBoost model** was trained on it. Instead of trying to model the entire complex signal, the model was tasked with the more focused job of finding predictive patterns within the residuals. The idea was that the XGBoost model could capture the complex, non-linear dynamics of the anomalies that were left over in the residual signal.

---

## 📈 Key Findings & Learnings

The results of this hybrid approach were not as effective as anticipated.

* **Poor Performance**: The STL-XGBoost model yielded poor results in detecting the anomalies accurately.
* **Lack of Pattern in Residuals**: A key reason for this was that the residual component did not show a clear, learnable pattern for the short-window anomalies characteristic of glass breakage. The decomposition, while technically sound, did not effectively isolate the anomaly signal in a way that the XGBoost model could easily learn.
* **Time Complexity**: Applying STL decomposition using a moving window across the entire dataset was also found to be computationally expensive and time-consuming.

This experiment was a valuable learning experience, demonstrating that while hybrid decomposition methods can be powerful, they are not universally effective. The negative result helped justify moving towards more advanced, end-to-end deep learning models that can learn to identify anomalies without the need for explicit signal decomposition.