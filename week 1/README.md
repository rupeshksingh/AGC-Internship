# Time Series Anomaly Detection Analysis (Week 1)

This project serves as an initial exploration into time series anomaly detection using an open-source dataset. The goal was to understand and implement various techniques for identifying anomalies in time-stamped data.

---

## 🎯 Objective

The primary objective of this initial analysis was to apply and evaluate different anomaly detection algorithms on a standard time series dataset. This foundational work was undertaken to build an understanding of the methods and challenges involved in detecting unusual patterns or outliers in sequential data.

---

## 📊 Dataset

The analysis was performed on the **Numenta Anomaly Benchmark (NAB)** dataset, specifically the `art_daily_jumpsup.csv` file. This dataset is designed to test the performance of anomaly detection algorithms and features a time series with sudden upward jumps.

---

## ⚙️ Analysis & Techniques

The notebook details a comprehensive analysis that includes data preprocessing, visualization, and the implementation of both statistical and machine learning-based anomaly detection methods.

* **Data Preprocessing**: The initial steps involved loading the data, converting the timestamp to a datetime object, and setting it as the index for time series analysis.
* **Visualization**: The time series data was plotted to visually inspect for any obvious anomalies or patterns.
* **Anomaly Detection Models**:
    * **Isolation Forest**: An unsupervised learning algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
    * **LSTM Autoencoder**: A neural network-based approach where a Long Short-Term Memory (LSTM) network is trained to reconstruct the time series data. Anomalies are identified where the reconstruction error is high.

---

## 📈 Key Findings

The analysis successfully implemented two distinct methods for anomaly detection. The LSTM Autoencoder, in particular, demonstrated a robust method for learning the normal patterns in the time series data and flagging deviations as anomalies based on reconstruction loss. This initial work provides a solid foundation for tackling more complex, domain-specific anomaly detection problems.
