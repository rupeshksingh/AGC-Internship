# Anomaly Detection using Isolation Forest & LSTM (Week 2)

This document outlines the benchmarking work performed using both an unsupervised machine learning model (Isolation Forest) and a deep learning model (LSTM) for time series anomaly detection on the glass rolling process dataset.

---

## 🎯 Objective

The goal of this phase was to establish baseline performance for anomaly detection using two different approaches:
1.  An **unsupervised** method (Isolation Forest) that does not require labeled data.
2.  A **supervised** method (LSTM) that learns from labeled anomaly sequences.

This allows for a comparison of how well each type of model can identify speed fluctuations indicative of glass breakage.

---

## 💾 Dataset

The analysis used the actual sensor data from the glass rolling process. Key preprocessing steps included:
* **Loading the Data**: Importing the time series data of the roller speeds.
* **Speed Binning**: Grouping the roller speeds into different bins to handle the various operating speeds corresponding to different glass thicknesses.
* **Sequence Creation**: For the LSTM model, the time series data was converted into sequences of a fixed length, which serve as the input for the network.

---

## ⚙️ Models & Methodology

### **Isolation Forest**
The Isolation Forest algorithm was implemented as an unsupervised approach. It works by isolating observations by randomly selecting a feature and then randomly selecting a split value. The key steps were:
* **Model Training**: An Isolation Forest model was trained on the speed data.
* **Prediction**: The model assigned an anomaly score to each data point. Points with a score of -1 were flagged as anomalies.
* **Evaluation**: The results were evaluated using a classification report and a confusion matrix to assess the model's performance in identifying anomalies.

### **Long Short-Term Memory (LSTM) Network**
A supervised deep learning approach was implemented using an LSTM network, which is well-suited for sequential data. The process was as follows:
* **Data Preparation**: The data was split into training and testing sets, and sequences were created.
* **Model Architecture**: A sequential LSTM model was built, consisting of LSTM layers followed by Dense layers.
* **Training**: The model was trained on the sequences to predict whether a sequence contained an anomaly.
* **Evaluation**: The model's performance was evaluated on the test set by plotting the training and validation loss and accuracy. Predictions were made on the test data, and the results were assessed.

---

## 📈 Key Findings

This benchmarking phase provided valuable insights into the performance of both unsupervised and supervised models on this specific time series dataset. The results from the classification reports and model evaluations serve as a baseline for comparing more advanced techniques in subsequent phases of the project. The LSTM, being a supervised model, was expected to perform well given the labeled data, while the Isolation Forest provided a quick, label-agnostic baseline.