import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
import gc
import warnings
warnings.filterwarnings('ignore')

def create_time_series_features(df, target_col='speed', has_label=True):
    """
    Create comprehensive time series features for anomaly detection
    """
    df = df.copy()

    lag_periods = [1]
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    rolling_windows = [30]
    for window in rolling_windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        # df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        # df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        # df[f'{target_col}_rolling_range_{window}'] = df[f'{target_col}_rolling_max_{window}'] - df[f'{target_col}_rolling_min_{window}']
        # df[f'{target_col}_rolling_skew_{window}'] = df[target_col].rolling(window=window).skew()
        # df[f'{target_col}_rolling_kurt_{window}'] = df[target_col].rolling(window=window).kurt()

    # diff_periods = [1]
    # for diff in diff_periods:
    #     df[f'{target_col}_diff_{diff}'] = df[target_col].diff(diff)
    #     df[f'{target_col}_diff_abs_{diff}'] = df[f'{target_col}_diff_{diff}'].abs()

    def calculate_trend(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                if not np.isnan(y).any():
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return slopes

    trend_windows = [30]
    for window in trend_windows:
        df[f'{target_col}_trend_{window}'] = calculate_trend(df[target_col], window)

    def rolling_entropy(series, window):
        entropy_values = []
        for i in range(len(series)):
            if i < window - 1:
                entropy_values.append(np.nan)
            else:
                window_data = series.iloc[i-window+1:i+1].values
                if not np.isnan(window_data).any():
                    hist, _ = np.histogram(window_data, bins=10)
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        ent = entropy(hist / hist.sum())
                        entropy_values.append(ent)
                    else:
                        entropy_values.append(np.nan)
                else:
                    entropy_values.append(np.nan)
        return entropy_values

    entropy_windows = [30]
    for window in entropy_windows:
        df[f'{target_col}_entropy_{window}'] = rolling_entropy(df[target_col], window)

    try:
        if len(df) > 51:
            df[f'{target_col}_savgol_filter'] = signal.savgol_filter(
                df[target_col].fillna(df[target_col].ffill()),
                window_length=51,
                polyorder=3
            )
        else:
            df[f'{target_col}_savgol_filter'] = df[target_col]
    except:
        df[f'{target_col}_savgol_filter'] = df[target_col]

    # for window in [30]:
    #     if len(df) > window:
    #         df[f'{target_col}_median_filter_{window}'] = signal.medfilt(
    #             df[target_col].fillna(df[target_col].ffill()),
    #             kernel_size=window if window % 2 == 1 else window + 1
    #         )

    try:
        b, a = signal.butter(4, 0.1)
        df[f'{target_col}_butterworth_filter'] = signal.filtfilt(
            b, a, df[target_col].fillna(df[target_col].ffill())
        )
    except:
        df[f'{target_col}_butterworth_filter'] = df[target_col]

    df[f'{target_col}_deviation_from_smooth'] = df[target_col] - df[f'{target_col}_savgol_filter']
    df[f'{target_col}_deviation_from_smooth_abs'] = df[f'{target_col}_deviation_from_smooth'].abs()

    for window in [30]:
        df[f'{target_col}_zscore_{window}'] = (
            (df[target_col] - df[f'{target_col}_rolling_mean_{window}']) /
            df[f'{target_col}_rolling_std_{window}']
        )

    df[f'{target_col}_acceleration'] = df[target_col].diff().diff()

    smallest_window = min(rolling_windows)
    df[f'{target_col}_to_rolling_mean_ratio'] = df[target_col] / df[f'{target_col}_rolling_mean_{smallest_window}']
    df[f'{target_col}_to_rolling_max_ratio'] = df[target_col] / df[f'{target_col}_rolling_max_{smallest_window}']

    if 'normalized' in target_col:
        df[f'{target_col}_extreme_high'] = (df[target_col] > 2).astype(int)
        df[f'{target_col}_extreme_low'] = (df[target_col] < -2).astype(int)

        df[f'{target_col}_cumsum'] = df[target_col].cumsum()

        for window in [30]:
            df[f'{target_col}_percentile_rank_{window}'] = (
                df[target_col].rolling(window=window)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            )

    return df

def create_sequences(X, y, seq_length):
    """
    Create sequences for TCNN input - simplified version
    """
    sequences = []
    labels = []

    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i:i + seq_length])
        # For each sequence, take the label at the last timestep
        labels.append(y[i + seq_length - 1])

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels

def build_simple_tcnn_model(seq_length, n_features):
    """
    Build a simplified TCNN model for binary classification
    """
    inputs = layers.Input(shape=(seq_length, n_features))

    # Simple TCN layers
    x = layers.Conv1D(32, 3, padding='causal', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(32, 3, padding='causal', dilation_rate=2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(32, 3, padding='causal', dilation_rate=4, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Global pooling to get a single vector
    x = layers.GlobalMaxPooling1D()(x)

    # Dense layers for classification
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output layer - single unit for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def train_anomaly_detector(data_path="Data/full_data.csv", seq_length=120):
    """
    Simplified training function for TCNN anomaly detection
    """
    # Clear any existing models from memory
    tf.keras.backend.clear_session()
    gc.collect()

    # Read data
    df = pd.read_csv(data_path, usecols=["speed", "pred_label"])

    start_datetime_str = "2025-05-13 10:15:18"
    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    df['time_value'] = [start_datetime + timedelta(seconds=i) for i in range(len(df))]

    # Binning and normalization
    bins = [20000, 23000, 26000, 30000, 34000, 38000, 40000, 42000, 44000, 46500, 50000, 55000, 60000, 65000, 70000, 83000]
    bins = sorted(list(set(bins)))
    df["speed_bin"] = pd.cut(df["speed"], bins=bins, labels=False)
    df["speed_bin"] = df["speed_bin"].astype(int)
    group_stats = df.groupby("speed_bin")["speed"].agg(["mean", "std"])
    df = df.merge(group_stats, how="left", left_on="speed_bin", right_index=True)
    df["speed_normalized"] = (df["speed"] - df["mean"]) / df["std"]

    print("Creating time series features...")
    df = create_time_series_features(df, target_col='speed_normalized')

    # Additional features
    for window in [120]:
        df[f'speed_rolling_mean_{window}'] = df['speed'].rolling(window=window).mean()
        df[f'speed_rolling_std_{window}'] = df['speed'].rolling(window=window).std()

    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['speed', 'pred_label', 'time_value', 'mean', 'std', 'speed_bin']]
    X = df[feature_cols].values
    y = df['pred_label'].values

    # Handle NaN values
    nan_mask = ~np.isnan(X).any(axis=1)
    X = X[nan_mask]
    y = y[nan_mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    print(f"Creating sequences with length {seq_length}...")
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

    # Split data
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Number of features: {X_train.shape[2]}")

    # Calculate class weight
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    if n_pos > 0:
        class_weight = {0: 1.0, 1: n_neg / n_pos}
    else:
        class_weight = {0: 1.0, 1: 1.0}

    print(f"Class distribution - Negative: {n_neg}, Positive: {n_pos}")
    print(f"Class weights: {class_weight}")

    # Build simplified model
    print("\nBuilding simplified TCNN model...")
    model = build_simple_tcnn_model(seq_length=seq_length, n_features=X_train.shape[2])

    # Compile with standard binary crossentropy
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print(f"Model parameters: {model.count_params():,}")

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    print("\nEvaluating model...")
    y_val_pred_proba = model.predict(X_val, batch_size=32, verbose=0).flatten()

    # Calculate metrics
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_avg_precision = average_precision_score(y_val, y_val_pred_proba)

    print(f"\nValidation AUC: {val_auc:.4f}")
    print(f"Validation Average Precision: {val_avg_precision:.4f}")

    # Classification report
    threshold = 0.5
    y_val_pred = (y_val_pred_proba > threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))

    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'bins': bins,
        'group_stats': group_stats,
        'seq_length': seq_length,
        'history': history
    }

def fill_anomaly_gaps(predictions, gap_threshold=10):
    """
    Fill gaps between anomaly segments if they are smaller than threshold
    """
    filled_predictions = predictions.copy()

    # Find anomaly segments
    anomaly_starts = []
    anomaly_ends = []

    in_anomaly = False
    for i in range(len(predictions)):
        if predictions[i] == 1 and not in_anomaly:
            anomaly_starts.append(i)
            in_anomaly = True
        elif predictions[i] == 0 and in_anomaly:
            anomaly_ends.append(i - 1)
            in_anomaly = False

    if in_anomaly:
        anomaly_ends.append(len(predictions) - 1)

    # Fill gaps
    for i in range(len(anomaly_ends) - 1):
        gap_start = anomaly_ends[i] + 1
        gap_end = anomaly_starts[i + 1] - 1
        gap_size = gap_end - gap_start + 1

        if gap_size <= gap_threshold:
            filled_predictions[gap_start:gap_end + 1] = 1

    return filled_predictions

def predict_anomalies(unlabeled_data, trained_model_dict, start_datetime_str="2025-05-13 10:15:18",
                     threshold=0.5, gap_threshold=10):
    """
    Predict anomalies on unlabeled data
    """
    model = trained_model_dict['model']
    scaler = trained_model_dict['scaler']
    feature_cols = trained_model_dict['feature_cols']
    bins = trained_model_dict['bins']
    group_stats = trained_model_dict['group_stats']
    seq_length = trained_model_dict['seq_length']

    # Prepare dataframe
    if isinstance(unlabeled_data, pd.Series):
        df = pd.DataFrame({'speed': unlabeled_data})
    elif isinstance(unlabeled_data, pd.DataFrame):
        df = unlabeled_data.copy()
        if 'A2:MCPGSpeed' in df.columns:
            df = df.rename(columns={'A2:MCPGSpeed': 'speed'})
    else:
        df = pd.DataFrame({'speed': unlabeled_data})

    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    df['time_value'] = [start_datetime + timedelta(seconds=i) for i in range(len(df))]

    # Binning and normalization
    bins = sorted(list(set(bins)))
    df["speed_bin"] = pd.cut(df["speed"], bins=bins, labels=False)
    df["speed_bin"] = df["speed_bin"].astype(int)
    df = df.merge(group_stats, how="left", left_on="speed_bin", right_index=True)
    df["speed_normalized"] = (df["speed"] - df["mean"]) / df["std"]

    # Create features
    df = create_time_series_features(df, target_col='speed_normalized', has_label=False)

    for window in [120]:
        df[f'speed_rolling_mean_{window}'] = df['speed'].rolling(window=window).mean()
        df[f'speed_rolling_std_{window}'] = df['speed'].rolling(window=window).std()

    X = df[feature_cols].values

    # Fill NaN values
    X = pd.DataFrame(X).ffill().bfill().values

    # Scale features
    X_scaled = scaler.transform(X)

    # Create sequences for prediction
    sequences = []
    for i in range(len(X_scaled) - seq_length + 1):
        sequences.append(X_scaled[i:i + seq_length])

    sequences = np.array(sequences)

    # Predict
    print(f"Predicting on {len(sequences)} sequences...")
    predictions_proba = model.predict(sequences, batch_size=32, verbose=1).flatten()

    # Create full predictions array
    full_predictions_proba = np.zeros(len(df))
    full_predictions_proba[:seq_length-1] = 0.0  # No predictions for initial points

    # Each prediction corresponds to the last timestep of each sequence
    for i in range(len(predictions_proba)):
        full_predictions_proba[i + seq_length - 1] = predictions_proba[i]

    # Apply threshold
    predictions = (full_predictions_proba > threshold).astype(int)

    # Fill gaps
    predictions_filled = fill_anomaly_gaps(predictions, gap_threshold)

    # Create results dataframe
    results = pd.DataFrame({
        'time_value': df['time_value'],
        'speed': df['speed'],
        'speed_normalized': df['speed_normalized'],
        'anomaly_prediction': predictions,
        'anomaly_prediction_filled': predictions_filled,
        'anomaly_probability': full_predictions_proba
    })

    # Calculate statistics
    n_anomalies_original = predictions.sum()
    n_anomalies_filled = predictions_filled.sum()

    print(f"\nTotal predictions: {len(results)}")
    print(f"Anomalies detected (original): {n_anomalies_original} ({n_anomalies_original/len(predictions)*100:.2f}%)")
    print(f"Anomalies detected (gap-filled): {n_anomalies_filled} ({n_anomalies_filled/len(predictions)*100:.2f}%)")

    return results

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Training TCNN anomaly detection model...")
    trained_model = train_anomaly_detector("full_data.csv", seq_length=40)

    print("\n" + "="*50)
    print("Making predictions on unlabeled data...")

    # Read unlabeled data
    try:
        unlabeled_df = pd.read_csv("input_data.csv", usecols=["A2:MCPGSpeed"])
    except:
        unlabeled_df = pd.read_csv("input_data.csv", usecols=["speed"])

    predictions = predict_anomalies(
        unlabeled_df,
        trained_model,
        threshold=0.3,
        gap_threshold=120
    )

    predictions.to_csv("anomaly_predictions_tcnn.csv", index=False)
    print(f"\nPredictions saved to 'anomaly_predictions_tcnn.csv'")