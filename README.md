# Time Series Anomaly Detection in the Glass Rolling Process

This repository contains the complete work for a data science internship project with **AGC Asia Pacific**. The project's goal was to develop a robust system for detecting anomalies in the glass rolling process using time series analysis and machine learning, culminating in a functional Streamlit application for model training and real-time inference.

## 📝 Problem Statement

In industrial glass manufacturing, the **glass rolling process** runs 24/7 and is critical for ensuring product quality. During this process, a sensor continuously monitors the speed of the rollers. When **glass breakage** occurs, it causes a distinct fluctuation in the roller speed.

The primary technical challenges were to:
* **Continuously monitor** high-frequency roller speed sensor data.
* **Detect the start and end** of anomaly sequences caused by speed fluctuations.
* Pinpoint the **"arrival point"** within an anomaly, where speed rises abruptly.

Solving this problem is critical as production downtime can cost **$10,000 to $50,000 per hour**, and defects impact customer satisfaction.

---

## 💾 Dataset

The dataset consists of high-frequency time series data from the roller speed sensors.
* **Data Source**: Sensor readings collected at **1-second intervals** from the manufacturing facility.
* **Characteristics**:
    * The dataset is **imbalanced**, with anomalies making up only about 13.4% of the data.
    * It contains multiple **operating speeds** due to different glass thicknesses being processed (from 2mm to 8mm).
    * Anomaly sequences have **variable lengths**, typically between 15 and 180 timesteps.
* **Preprocessing**: To handle the different operating speeds, the speed data was **binned**, and features were engineered specifically for each bin.

---

## 🛠️ Project Workflow & Methodologies

The project was executed in an iterative fashion, progressing from simple baselines to a state-of-the-art deep learning architecture.

### **Phase 1: Baseline Models & Research Exploration**
The initial phase focused on establishing performance baselines and exploring recent research models.
* **ML & DL Baselines**: Standard models like **Isolation Forest, XGBoost, and LSTMs** were tested, achieving accuracies around **80%**.
* **Research Models (DADA & CATCH)**: Pre-trained, state-of-the-art models were implemented.
    * **DADA** (Dual Adversarial Decoders) achieved **82% accuracy**.
    * **CATCH** (Channel-Aware multivariate Time series detection) achieved **84% accuracy**.
    These models showed promise but required significant data formatting and adaptation.

### **Phase 2: XGBoost with Feature Engineering**
A deep dive was conducted using an XGBoost model with over 50 engineered features (rolling stats, FFT, temporal features).
* **Initial Success**: The model showed a promising **86% test accuracy**.
* **Critical Learning (Data Leakage)**: The model failed on new production data. A thorough review revealed that **data leakage** occurred during feature engineering, as rolling windows inadvertently included future information.
* **Honest Evaluation**: After implementing strict temporal validation and removing look-ahead features, the corrected accuracy was **78%**. This was a crucial lesson on the importance of rigorous, time-aware validation.

### **Phase 3: Hybrid & Ensemble Approaches**
With learnings from the previous phases, more complex hybrid strategies were tested.
* **STL Decomposition + XGBoost**: The time series was decomposed into trend, seasonal, and residual components. An XGBoost model trained on the residuals performed poorly, as the anomaly patterns were not clearly isolated in the residual signal.
* **Dual Training Approach**: This novel approach used two specialist models:
    1.  An **"Anomaly Expert"** trained on anomaly-rich data.
    2.  A **"Normality Expert"** trained exclusively on normal data.
    The final prediction was a weighted combination of their outputs. This method was highly effective, achieving **91% accuracy** after post-processing.

### **Phase 4: Final Model - TCNN with TENET Framework**
The final and most successful model was a **Temporal Convolutional Neural Network (TCNN)** with an attention mechanism, based on the TENET framework.
* **Architecture**: The TCNN used **dilated causal convolutions** to capture long-range dependencies, while the **attention mechanism** helped it focus on the most important timesteps for prediction.
* **Features**: A carefully selected set of **15 advanced features** (statistical, temporal, and frequency-domain) was used.
* **Training**: The model used a `BinaryFocalCrossEntropy` loss function to handle the severe class imbalance.
* **Post-Processing**: The raw predictions were refined by removing short anomaly sequences and filling small gaps, which was critical for real-world performance.

---

## 📊 Model Performance Summary

| Model / Approach | Final Accuracy |
| :--- | :---: |
| Baseline ML Models | 80%  |
| Enhanced XGBoost | 83%  |
| Research Models (DADA, CATCH) | 86%  |
| Dual Training Approach | 91%  |
| **TCNN - TENET (Final Model)** | **99%**  |

The TCNN model was highly efficient, with an inference time of **470ms per sequence** and a small **2MB model size**, making it ideal for deployment.

---

## 🚀 Final Deliverable: Streamlit Application

The project concluded with the development of an interactive Streamlit web application (`app.py`). This application provides a user-friendly GUI to demonstrate the project's capabilities.

**Key Features**:
* **Model Selection**: Allows users to choose between the **XGBoost** and the final **TCNN (TENET)** model.
* **Dual Modes**: Supports both **Training** on new data and **Real-Time Inference** on uploaded files.
* **Interactive Visualization**: Plots the time series data and clearly highlights the detected anomalies, providing an intuitive way to see the models in action.

---

## 💡 Key Learnings

* **Temporal Validation is Critical**: The experience with data leakage in the XGBoost model highlighted that proper, time-aware validation is non-negotiable in time series projects to avoid misleading results.
* **Feature Quality > Quantity**: The final TCNN model performed best with a curated set of 15 advanced features, proving that thoughtful feature engineering is more effective than using a vast number of simple features.
* **Attention Mechanisms for Sequences**: The success of the TCNN with attention confirmed that these architectures are exceptionally powerful for sequence anomaly detection tasks.

---

## 🙏 Acknowledgments

I would like to extend my sincere gratitude to the **Data Science Team** at AGC Asia Pacific, my Internship Supervisor **Roushan Sir**, and my Technical Mentor **Rishit Sir** for their invaluable guidance and support throughout this project.