import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import entropy, iqr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import gc
import warnings

warnings.filterwarnings('ignore')

def rolling_fft_features(series, window):
    spectral_centroids, spectral_entropies = [], []
    for i in range(len(series)):
        if i < window - 1:
            spectral_centroids.append(np.nan)
            spectral_entropies.append(np.nan)
        else:
            window_data = series.iloc[i-window+1:i+1].values
            if not np.isnan(window_data).any():
                fft_vals, fft_freq = np.fft.rfft(window_data), np.fft.rfftfreq(len(window_data))
                fft_mag = np.abs(fft_vals)
                centroid = np.sum(fft_mag * fft_freq) / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0
                spectral_centroids.append(centroid)
                psd = fft_mag**2
                psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
                spec_entropy = entropy(psd_norm) if len(psd_norm[psd_norm > 0]) > 0 else 0
                spectral_entropies.append(spec_entropy)
            else:
                spectral_centroids.append(np.nan)
                spectral_entropies.append(np.nan)
    return spectral_centroids, spectral_entropies

def create_time_series_features_advanced(df, target_col='speed_normalized'):
    df = df.copy()
    window = 70
    df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
    df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    df[f'{target_col}_rolling_iqr_{window}'] = df[target_col].rolling(window=window).apply(iqr, raw=True)
    alpha = 0.95
    df[f'{target_col}_ewm_mean_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
    df[f'{target_col}_ewm_std_{alpha}'] = df[target_col].ewm(alpha=alpha).std()
    diff_period = 70
    df[f'{target_col}_diff_abs_{diff_period}'] = df[target_col].diff(diff_period).abs()
    trend_window = 70
    slopes = [np.polyfit(np.arange(trend_window), df[target_col].iloc[i-trend_window+1:i+1], 1)[0]
              if i >= trend_window - 1 and not df[target_col].iloc[i-trend_window+1:i+1].isnull().any()
              else np.nan for i in range(len(df))]
    df[f'{target_col}_trend_{trend_window}'] = slopes
    df[f'{target_col}_acceleration'] = df[target_col].diff().diff()
    fft_window = 70
    spectral_centroids, spectral_entropies = rolling_fft_features(df[target_col], fft_window)
    df[f'{target_col}_fft_spectral_centroid_{fft_window}'] = spectral_centroids
    df[f'{target_col}_fft_spectral_entropy_{fft_window}'] = spectral_entropies
    df[f'{target_col}_zscore_{window}'] = (df[target_col] - df[f'{target_col}_rolling_mean_{window}']) / (df[f'{target_col}_rolling_std_{window}'] + 1e-6)
    df[f'{target_col}_to_rolling_mean_ratio'] = df[target_col] / (df[f'{target_col}_rolling_mean_{window}'] + 1e-6)
    df[f'{target_col}_to_rolling_max_ratio'] = df[target_col] / (df[f'{target_col}_rolling_max_{window}'] + 1e-6)
    return df

def create_sequences(X, y, seq_length):
    sequences, labels = [], []
    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i:i + seq_length])
        labels.append(y[i + seq_length - 1])
    return np.array(sequences), np.array(labels)

class TemporalResidualBlock(layers.Layer):
    def __init__(self, n_filters, kernel_size, dilation_rate, **kwargs):
        super(TemporalResidualBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.conv1 = layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, padding='causal', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, padding='causal', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.downsample = None

    def build(self, input_shape):
        if input_shape[-1] != self.n_filters:
            self.downsample = layers.Conv1D(filters=self.n_filters, kernel_size=1,
                                            padding='same')
        super(TemporalResidualBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.bn1(inputs)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)

        res = inputs
        if self.downsample is not None:
            res = self.downsample(res)

        return layers.add([x, res])

class AttentionBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = tf.cast(input_shape[-1], tf.float32) ** -0.5
        super(AttentionBlock, self).build(input_shape)

    def call(self, inputs):
        q = inputs
        k = inputs
        v = inputs

        scores = tf.linalg.matmul(q, k, transpose_b=True) * self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        attention = tf.linalg.matmul(weights, v)

        return attention

def build_tenet_model(seq_length, n_features):
    """
    Builds a model based on the TENET framework (TCNA architecture).
    """
    inputs = layers.Input(shape=(seq_length, n_features))

    n_filters = 64
    kernel_size = 7
    num_blocks = 4

    x = inputs

    for i in range(num_blocks):
        dilation_rate = 2**i
        x = TemporalResidualBlock(n_filters=n_filters, kernel_size=kernel_size,
                                  dilation_rate=dilation_rate)(x)

    x = AttentionBlock()(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def train_anomaly_detector(data_path="full_data.csv", seq_length=70):
    tf.keras.backend.clear_session()
    gc.collect()

    df = pd.read_csv(data_path, usecols=["speed", "pred_label"])

    bins = [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]
    df["speed_bin"] = pd.cut(df["speed"], bins=bins, labels=False, include_lowest=True)
    df["speed_bin"] = df["speed_bin"].astype(int)
    group_stats = df.groupby("speed_bin")["speed"].agg(["mean", "std"])
    df = df.merge(group_stats, how="left", left_on="speed_bin", right_index=True)
    df["speed_normalized"] = (df["speed"] - df["mean"]) / df["std"]
    df["speed_normalized"].fillna(0, inplace=True)

    print("Creating advanced time series features...")
    df_features = create_time_series_features_advanced(df, target_col='speed_normalized')
    df_features['speed_normalized_raw'] = df_features['speed_normalized']
    feature_cols = [col for col in df_features.columns if col not in ['speed', 'pred_label', 'time_value', 'mean', 'std', 'speed_bin', 'speed_normalized']]

    X = df_features[feature_cols].values
    y = df_features['pred_label'].values
    X = pd.DataFrame(X, columns=feature_cols).ffill().bfill().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

    split_idx = int(len(X_seq) * 0.9)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")

    print("\nBuilding TENET-based model...")
    model = build_tenet_model(seq_length=seq_length, n_features=X_train.shape[2])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.8),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision')]
    )
    model.summary()
    print(f"Model parameters: {model.count_params():,}")

    early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max', verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("\nEvaluating model...")
    y_val_pred_proba = model.predict(X_val, batch_size=64, verbose=0).flatten()
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_avg_precision = average_precision_score(y_val, y_val_pred_proba)
    print(f"\nValidation AUC: {val_auc:.4f}")
    print(f"Validation Average Precision (PR-AUC): {val_avg_precision:.4f}")

    threshold = 0.5
    y_val_pred = (y_val_pred_proba > threshold).astype(int)
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y_val, y_val_pred, target_names=['Normal', 'Anomaly']))

    return {'model': model, 'scaler': scaler, 'feature_cols': feature_cols, 'bins': bins, 'group_stats': group_stats, 'seq_length': seq_length, 'history': history, 'threshold': threshold}

def predict_anomalies(unlabeled_data, trained_model_dict, start_datetime_str="2025-05-13 10:15:18"):
    model = trained_model_dict['model']
    scaler = trained_model_dict['scaler']
    feature_cols = trained_model_dict['feature_cols']
    bins = trained_model_dict['bins']
    group_stats = trained_model_dict['group_stats']
    seq_length = trained_model_dict['seq_length']
    threshold = trained_model_dict['threshold']

    if isinstance(unlabeled_data, pd.Series):
        df = pd.DataFrame({'speed': unlabeled_data})
    else:
        df = unlabeled_data.copy()
        if 'A2:MCPGSpeed' in df.columns: df = df.rename(columns={'A2:MCPGSpeed': 'speed'})
    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    df['time_value'] = [start_datetime + timedelta(seconds=i) for i in range(len(df))]
    df["speed_bin"] = pd.cut(df["speed"], bins=bins, labels=False, include_lowest=True)
    df = df.merge(group_stats, how="left", left_on="speed_bin", right_index=True)
    df["speed_normalized"] = (df["speed"] - df["mean"]) / df["std"]
    df["speed_normalized"].fillna(0, inplace=True)

    df_features = create_time_series_features_advanced(df, target_col='speed_normalized')
    df_features['speed_normalized_raw'] = df_features['speed_normalized']
    X = df_features[feature_cols].values
    X = pd.DataFrame(X, columns=feature_cols).ffill().bfill().values
    X_scaled = scaler.transform(X)

    sequences = []
    for i in range(len(X_scaled) - seq_length + 1):
        sequences.append(X_scaled[i:i + seq_length])
    sequences = np.array(sequences)

    print(f"Predicting on {len(sequences)} sequences...")
    predictions_proba = model.predict(sequences, batch_size=64, verbose=1).flatten()

    full_predictions_proba = np.zeros(len(df))
    full_predictions_proba[:seq_length-1] = 0.0
    for i in range(len(predictions_proba)):
        full_predictions_proba[i + seq_length - 1] = predictions_proba[i]

    predictions_raw = (full_predictions_proba > threshold).astype(int)
    predictions_cleaned = predictions_raw

    results = pd.DataFrame({
        'time_value': df['time_value'], 'speed': df['speed'],
        'anomaly_prediction': predictions_cleaned,
        'anomaly_prediction_filled': predictions_cleaned,
        'anomaly_probability': full_predictions_proba
    })

    print(f"\nAnomalies detected (cleaned): {predictions_cleaned.sum()} ({predictions_cleaned.sum()/len(predictions_cleaned)*100:.2f}%)")
    return results

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Training TENET-based anomaly detection model...")
    trained_model = train_anomaly_detector("full_data.csv", seq_length=70)

    print("\n" + "="*50)
    print("Making predictions on unlabeled data...")
    try:
        unlabeled_df = pd.read_csv("input_data.csv", usecols=["A2:MCPGSpeed"])
    except FileNotFoundError:
        print("Could not find 'input_data.csv' with 'A2:MCPGSpeed', trying 'speed' column.")
        unlabeled_df = pd.read_csv("input_data.csv", usecols=["speed"])

    predictions = predict_anomalies(
        unlabeled_df,
        trained_model,
    )

    predictions.to_csv("anomaly_predictions_tenet.csv", index=False)
    print(f"\nPredictions saved to 'anomaly_predictions_tenet.csv'")