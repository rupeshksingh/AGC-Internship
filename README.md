# Time Series Anomaly Detection in the Glass Rolling Process

[cite_start]This repository contains the complete work for a data science internship project with **AGC Asia Pacific**[cite: 12, 39]. The project's goal was to develop a robust system for detecting anomalies in the glass rolling process using time series analysis and machine learning, culminating in a functional Streamlit application for model training and real-time inference.

## 📝 Problem Statement

[cite_start]In industrial glass manufacturing, the **glass rolling process** runs 24/7 and is critical for ensuring product quality[cite: 46, 48]. During this process, a sensor continuously monitors the speed of the rollers. When **glass breakage** occurs, it causes a distinct fluctuation in the roller speed.

The primary technical challenges were to:
* [cite_start]**Continuously monitor** high-frequency roller speed sensor data[cite: 52, 53].
* [cite_start]**Detect the start and end** of anomaly sequences caused by speed fluctuations[cite: 54, 57].
* Pinpoint the **"arrival point"** within an anomaly, where speed rises abruptly.

[cite_start]Solving this problem is critical as production downtime can cost **$10,000 to $50,000 per hour**, and defects impact customer satisfaction[cite: 65, 66].

---

## 💾 Dataset

The dataset consists of high-frequency time series data from the roller speed sensors.
* [cite_start]**Data Source**: Sensor readings collected at **1-second intervals** from the manufacturing facility[cite: 94].
* **Characteristics**:
    * [cite_start]The dataset is **imbalanced**, with anomalies making up only about 13.4% of the data[cite: 94].
    * [cite_start]It contains multiple **operating speeds** due to different glass thicknesses being processed (from 2mm to 8mm)[cite: 47, 94].
    * [cite_start]Anomaly sequences have **variable lengths**, typically between 15 and 180 timesteps[cite: 94].
* **Preprocessing**: To handle the different operating speeds, the speed data was **binned**, and features were engineered specifically for each bin.

---

## 🛠️ Project Workflow & Methodologies

The project was executed in an iterative fashion, progressing from simple baselines to a state-of-the-art deep learning architecture.

### **Phase 1: Baseline Models & Research Exploration**
The initial phase focused on establishing performance baselines and exploring recent research models.
* [cite_start]**ML & DL Baselines**: Standard models like **Isolation Forest, XGBoost, and LSTMs** were tested, achieving accuracies around **80%**[cite: 260].
* **Research Models (DADA & CATCH)**: Pre-trained, state-of-the-art models were implemented.
    * [cite_start]**DADA** (Dual Adversarial Decoders) achieved **82% accuracy**[cite: 148].
    * [cite_start]**CATCH** (Channel-Aware multivariate Time series detection) achieved **84% accuracy**[cite: 148].
    These models showed promise but required significant data formatting and adaptation.

### **Phase 2: XGBoost with Feature Engineering**
A deep dive was conducted using an XGBoost model with over 50 engineered features (rolling stats, FFT, temporal features).
* **Initial Success**: The model showed a promising **86% test accuracy**.
* **Critical Learning (Data Leakage)**: The model failed on new production data. A thorough review revealed that **data leakage** occurred during feature engineering, as rolling windows inadvertently included future information.
* [cite_start]**Honest Evaluation**: After implementing strict temporal validation and removing look-ahead features, the corrected accuracy was **78%**[cite: 163]. This was a crucial lesson on the importance of rigorous, time-aware validation.

### **Phase 3: Hybrid & Ensemble Approaches**
With learnings from the previous phases, more complex hybrid strategies were tested.
* **STL Decomposition + XGBoost**: The time series was decomposed into trend, seasonal, and residual components. An XGBoost model trained on the residuals performed poorly, as the anomaly patterns were not clearly isolated in the residual signal.
* **Dual Training Approach**: This novel approach used two specialist models:
    1.  An **"Anomaly Expert"** trained on anomaly-rich data.
    2.  A **"Normality Expert"** trained exclusively on normal data.
    The final prediction was a weighted combination of their outputs. [cite_start]This method was highly effective, achieving **91% accuracy** after post-processing[cite: 258].

### **Phase 4: Final Model - TCNN with TENET Framework**
[cite_start]The final and most successful model was a **Temporal Convolutional Neural Network (TCNN)** with an attention mechanism, based on the TENET framework[cite: 227].
* [cite_start]**Architecture**: The TCNN used **dilated causal convolutions** to capture long-range dependencies, while the **attention mechanism** helped it focus on the most important timesteps for prediction[cite: 229].
* [cite_start]**Features**: A carefully selected set of **15 advanced features** (statistical, temporal, and frequency-domain) was used[cite: 243, 245].
* [cite_start]**Training**: The model used a `BinaryFocalCrossEntropy` loss function to handle the severe class imbalance[cite: 303].
* [cite_start]**Post-Processing**: The raw predictions were refined by removing short anomaly sequences and filling small gaps, which was critical for real-world performance[cite: 267, 269].

---

## 📊 Model Performance Summary

| Model / Approach | Final Accuracy |
| :--- | :---: |
| Baseline ML Models | [cite_start]80% [cite: 260] |
| Enhanced XGBoost | [cite_start]83% [cite: 262] |
| Research Models (DADA, CATCH) | [cite_start]86% [cite: 261] |
| Dual Training Approach | [cite_start]91% [cite: 258] |
| **TCNN - TENET (Final Model)** | [cite_start]**99%** [cite: 259] |

[cite_start]The TCNN model was highly efficient, with an inference time of **470ms per sequence** and a small **2MB model size**, making it ideal for deployment[cite: 305].

---

## 🚀 Final Deliverable: Streamlit Application

The project concluded with the development of an interactive Streamlit web application (`app.py`). This application provides a user-friendly GUI to demonstrate the project's capabilities.

**Key Features**:
* **Model Selection**: Allows users to choose between the **XGBoost** and the final **TCNN (TENET)** model.
* **Dual Modes**: Supports both **Training** on new data and **Real-Time Inference** on uploaded files.
* **Interactive Visualization**: Plots the time series data and clearly highlights the detected anomalies, providing an intuitive way to see the models in action.

---

## 💡 Key Learnings

* [cite_start]**Temporal Validation is Critical**: The experience with data leakage in the XGBoost model highlighted that proper, time-aware validation is non-negotiable in time series projects to avoid misleading results[cite: 321].
* [cite_start]**Feature Quality > Quantity**: The final TCNN model performed best with a curated set of 15 advanced features, proving that thoughtful feature engineering is more effective than using a vast number of simple features[cite: 319].
* [cite_start]**Attention Mechanisms for Sequences**: The success of the TCNN with attention confirmed that these architectures are exceptionally powerful for sequence anomaly detection tasks[cite: 318].

---

## 🙏 Acknowledgments

[cite_start]I would like to extend my sincere gratitude to the **Data Science Team** at AGC Asia Pacific, my Internship Supervisor **Roushan Sir**, and my Technical Mentor **Rishit Sir** for their invaluable guidance and support throughout this project[cite: 333].