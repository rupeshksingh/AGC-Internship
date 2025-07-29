# Final Deliverable: Streamlit App for Anomaly Detection (Week 10)

This document describes the final deliverable of the project, completed in the tenth week. An interactive Streamlit web application was developed to provide a user-friendly interface for training and running real-time inference with the project's key models.

---

## 🎯 Objective

The goal for the final week was to bridge the gap between model development and practical application. The objective was to create an accessible and interactive tool that allows users to:
* **Train** the anomaly detection models on new data.
* **Perform real-time inference** to detect anomalies.
* **Visualize** the results in an intuitive way.
* **Compare** the performance of the final two best models: the **XGBoost** model and the **TENET (TCNN)** model.

---

## 🚀 Application Overview

The final deliverable is a web application built using the **Streamlit** framework, as detailed in the `app.py` script. This application serves as a comprehensive graphical user interface (GUI) for the entire project, showcasing its core functionalities in a clear and interactive manner.

---

## ✨ Key Features

The application is designed with a simple and intuitive workflow, allowing users to easily select a model and a mode of operation.

### **1. Model Selection**
The user can choose which model to work with from a dropdown menu:
* **XGBoost Model**: The enhanced gradient boosting model with extensive feature engineering.
* **TENET (TCNN) Model**: The final, state-of-the-art deep learning model that delivered the best performance.

### **2. Mode of Operation**
The application supports two primary modes:
* **Training Mode**: This mode allows a user to upload their own time series data. The app then uses this data to train the selected model from scratch, displaying progress and final performance metrics upon completion.
* **Inference Mode**: In this mode, a user can upload a data file for anomaly detection. The app processes the data with the chosen pre-trained model and displays a plot of the time series, clearly highlighting the detected anomalies in real-time.

### **3. Interactive Visualization**
The application provides clear and interactive visualizations of the results. In inference mode, the time series is plotted with anomalies marked, allowing for easy interpretation of the model's output, similar to the final results plot from the TCNN model.

---

## 🛠️ Technical Stack

The application leverages a number of key Python libraries:
* **Web Framework**: Streamlit
* **Data Handling**: Pandas, NumPy
* **Machine Learning**: Scikit-learn, XGBoost
* **Deep Learning**: TensorFlow/Keras
* **Visualization**: Plotly, Matplotlib

This Streamlit application serves as the perfect conclusion to the project, transforming the complex models into a tangible and usable tool that effectively demonstrates the success of the developed anomaly detection solution.