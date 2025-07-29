# Final Model: TCNN with Attention for Anomaly Detection (Week 9)

This document details the final and most successful modeling phase of the project, conducted in the ninth week. This work involved implementing a **Temporal Convolutional Neural Network (TCNN)** with an attention mechanism, which yielded outstanding results in detecting anomalies in the glass rolling process.

---

## 🎯 Objective

The objective for this final week was to leverage a state-of-the-art, end-to-end deep learning architecture to achieve the highest possible accuracy and robustness. The goal was to build a model that could effectively learn complex temporal dependencies from the time series data without the need for complex hybrid approaches or manual decomposition.

---

## ⚙️ Methodology: TCNN & TENET Framework

[cite_start]The final solution, implemented in the `tcnn.py` script, is based on the **TENET framework**, which uses a TCNN with an attention mechanism[cite: 228].

### **Architecture**
* [cite_start]**Temporal Convolutional Network (TCNN)**: The core of the model is a TCNN, which uses **dilated causal convolutions**[cite: 229]. [cite_start]This allows the model to have a very large receptive field, enabling it to capture long-range patterns and dependencies in the time series data efficiently[cite: 229].
* [cite_start]**Attention Mechanism**: An attention mechanism is integrated into the network to help the model weigh the importance of different timesteps within a sequence[cite: 229]. [cite_start]This allows it to focus on the most critical moments when making a prediction, further improving accuracy[cite: 229].

### **Final Feature Engineering**
[cite_start]A carefully selected set of **15 optimized features** was used to feed the model[cite: 243]. [cite_start]These features were designed to capture a broad range of information while remaining efficient[cite: 244]. The feature set included:
* [cite_start]**Statistical Features**: Metrics like `rolling_iqr` (inter-quartile range), `ewm_std` (exponentially weighted moving standard deviation), and `z_scores`[cite: 247].
* [cite_start]**Temporal Features**: Features that capture trends over time, such as `slopes` and other trend-related metrics[cite: 249].
* [cite_start]**Frequency-Domain Features**: Information extracted from the spectral representation of the signal, including `FFT components`, `spectral_entropy`, and `spectral_centroid`[cite: 251].

### **Training Configuration**
To handle the class imbalance and train the model effectively, the following configuration was used:
* [cite_start]**Loss Function**: `BinaryFocalCrossEntropy` was chosen to focus the model's training on hard-to-classify examples, which is ideal for imbalanced datasets[cite: 303].
* [cite_start]**Optimizer**: The Adam optimizer was used with learning rate scheduling[cite: 303].
* [cite_start]**Training Time**: The model was trained for 50 epochs with a batch size of 32, taking approximately 30 minutes on a GPU[cite: 303].

---

## ✨ Post-Processing & Results

[cite_start]To refine the raw predictions and make them more practical for real-world application, several post-processing steps were applied[cite: 273].

### **Refinement Steps**
* [cite_start]Filtered out very short anomaly detections (sequences shorter than 30 timesteps) to reduce false positives[cite: 267, 268].
* [cite_start]Filled small gaps between detected anomaly sequences to capture the full event[cite: 269, 270].
* [cite_start]Applied temporal smoothing using a majority vote within a sliding window to stabilize the final predictions[cite: 271, 272].

### **Performance**
* [cite_start]**Accuracy**: The final TCNN model achieved an outstanding **99% accuracy**, outperforming all previous models[cite: 259, 264].
* [cite_start]**Effectiveness**: Visual inspection of the results showed that the model **perfectly recognized all 5 anomaly sequences** in a sample test case, demonstrating its high precision and recall[cite: 297].
* [cite_start]**Deployment Readiness**: The model is highly efficient, with a real-time inference time of just **470ms per sequence** and a small memory footprint of **2 MB**, making it well-suited for integration into a production environment[cite: 305].