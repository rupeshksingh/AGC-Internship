# Mid-Term Review & XGBoost Deep Dive (Week 6)

This document outlines the work from the sixth week, which was centered around a mid-term presentation to stakeholders and a comprehensive Exploratory Data Analysis (EDA) of the enhanced XGBoost model's predictions and features.

---

## 🎯 Objective

This week marked a crucial checkpoint in the project. The primary goals were to:
1.  [cite_start]**Present Project Progress**: To deliver a mid-term presentation summarizing the problem, methodologies explored, key learnings, and current results to the project supervisors and team[cite: 2].
2.  **Analyze XGBoost Performance**: To conduct a deep-dive EDA into the behavior of the enhanced XGBoost model to understand its predictions, feature importances, and overall performance in detail.

---

## 🎤 Mid-Term Presentation

A presentation was prepared and delivered to communicate the project's progress.

* **Content**: The presentation covered the entire project lifecycle up to this point, including:
    * [cite_start]An overview of the **problem statement** and the business impact of anomaly detection in the glass rolling process[cite: 2].
    * [cite_start]A summary of the **methodologies explored**, from initial baselines (Isolation Forest, LSTM) to advanced research models (DADA, CATCH)[cite: 2].
    * [cite_start]A detailed section on the **XGBoost model**, including the feature engineering process and the critical lesson learned about **data leakage** and the need for proper temporal validation[cite: 2].
    * [cite_start]A review of the current results and the plan for the final phase of the project, which involved exploring more advanced architectures like TCNNs[cite: 2].

---

## 🔎 EDA of XGBoost Model & Features

A detailed EDA was performed to better understand the inner workings of the optimized XGBoost model.

* **Methodology**: The analysis, documented in the `eda_xgb.ipynb` notebook, focused on several key areas:
    * **Feature Importance**: Plotted the top features to identify which characteristics of the data were most influential in the model's predictions. This helps in understanding what signals the model is picking up on to detect anomalies.
    * **Probability Distribution Analysis**: Visualized the distribution of the predicted probabilities for both normal and anomalous classes to assess the model's confidence and its ability to separate the two classes.
    * **Calibration Plot**: Created a calibration plot to check if the model's predicted probabilities are reliable. This is important for trusting the model's output in a production environment.
    * **Rolling Performance**: Analyzed the model's performance over time on a rolling basis to ensure its predictions are stable and consistent across different parts of the dataset.

* **Findings**: The EDA provided a much deeper understanding of the XGBoost model's strengths and weaknesses. It confirmed that the model was learning relevant patterns from the engineered features and provided valuable insights into its predictive behavior, which helped validate its performance before moving on to the final project phase.