# Advanced Anomaly Detection: VLM, DADA, and Feature Engineering (Week 3)

This document outlines the work from the third week, which involved exploring state-of-the-art and feature-intensive methods for anomaly detection. The objective was to test the limits of current research models and feature engineering techniques on the glass rolling process dataset.

---

## 🎯 Objective

The goal for this week was to move beyond baseline models and investigate three distinct and advanced methodologies:
1.  **Vision-Language Models (VLM)**: To test a novel approach by converting time series data into images for anomaly classification.
2.  **DADA Model**: To implement a recent unsupervised, pre-trained research model designed for robust anomaly detection.
3.  **XGBoost with Extensive Feature Engineering**: To build a powerful classifier by creating a comprehensive set of features and understanding the impact of proper validation.

---

## ⚙️ Methodologies & Findings

### **1. Vision-Language Models (VLM) for Anomaly Detection**
This unconventional approach treated anomaly detection as an image classification problem.

* **Methodology**:
    * Time series segments were converted into images using the **Gramian Angular Summation Field (GASF)** technique.
    * A pre-trained **Vision Transformer (ViT)** model was then fine-tuned on these generated images to classify them as either "Normal" or "Anomaly".
* **Findings**:
    * This method provides an innovative way to capture temporal patterns and dependencies visually.
    * The implementation required significant preprocessing to transform the sequential data into a 2D format suitable for vision models. It served as a creative proof-of-concept for applying vision models to time series data.

### **2. DADA Model Implementation**
The **DADA (Dual Adversarial Decoders with Adaptive bottlenecks)** model, a recent advancement in unsupervised anomaly detection, was tested on the dataset.

* **Methodology**:
    * The pre-trained DADA model, which was trained on a diverse set of multi-domain time series data, was used for zero-shot anomaly detection.
    * The glass rolling speed data was formatted to be compatible with the model's input requirements. The model then provided anomaly scores for the time series.
* **Findings**:
    * [cite_start]DADA demonstrated strong out-of-the-box performance without needing to be retrained on the specific dataset. [cite: 141]
    * [cite_start]This phase highlighted the potential of using large, pre-trained models for industrial applications, though it also brought up challenges related to data formatting and adapting the model to a specific domain. [cite: 146]

### **3. XGBoost with Feature Engineering**
This approach focused on building a highly accurate classifier by manually engineering a wide array of features.

* **Methodology**:
    * A large set of over 50 features was created, capturing different aspects of the time series, including:
        * **Rolling Statistics**: `mean`, `std`, `min`, `max`, `IQR`.
        * **Temporal Features**: `difference`, `slope`, `ewm` (Exponentially Weighted Moving Averages).
        * **Frequency Domain Features**: `FFT` components.
    * An **XGBoost** model was trained on these features to classify each time point.
* **Critical Learning: Data Leakage**:
    * [cite_start]The model initially showed a very promising **86% accuracy**. [cite: 161] However, it performed poorly on new, unseen production data.
    * [cite_start]A thorough review revealed that **data leakage** had occurred during feature engineering. [cite: 161] Features like rolling means were calculated using windows that included future data points, which the model inadvertently learned from.
    * [cite_start]After implementing strict temporal validation and removing look-ahead features, the model's "honest" accuracy was found to be **78%**. [cite: 162, 163] [cite_start]This was a critical lesson on the importance of rigorous, time-aware validation in time series forecasting and classification tasks. [cite: 163]