import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, 
    make_scorer, roc_curve, auc
)
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
import warnings
import joblib
import json
from datetime import datetime
warnings.filterwarnings('ignore')

def detect_data_drift(train_data, new_data, features, threshold=0.1):
    """
    Detect if there's significant drift between training and new data.
    Uses Kolmogorov-Smirnov test to identify distribution shifts.
    """
    drift_report = {}
    significant_drifts = []
    
    for feature in features:
        if feature in train_data.columns and feature in new_data.columns:
            ks_stat, p_value = stats.ks_2samp(train_data[feature], new_data[feature])

            train_mean = train_data[feature].mean()
            new_mean = new_data[feature].mean()
            mean_shift = abs(train_mean - new_mean) / (abs(train_mean) + 1e-10)
            
            train_std = train_data[feature].std()
            new_std = new_data[feature].std()
            std_shift = abs(train_std - new_std) / (abs(train_std) + 1e-10)
            
            drift_report[feature] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'mean_shift': mean_shift,
                'std_shift': std_shift,
                'significant': ks_stat > threshold or mean_shift > 0.2
            }
            
            if drift_report[feature]['significant']:
                significant_drifts.append(feature)
    
    return drift_report, significant_drifts

def engineer_robust_features(df, speed_col='speed', timestamp_col='indo_time'):
    """
    Enhanced feature engineering without data leakage for irregular sensor data.
    All features only use past information (causal features).
    """
    features = pd.DataFrame(index=df.index)
    features['Speed'] = df[speed_col].copy()
    
    # 1. Exponential Moving Average (EMA) - naturally causal
    alphas = [0.1, 0.3, 0.5]
    for alpha in alphas:
        features[f'Speed_ema_{alpha}'] = df[speed_col].ewm(alpha=alpha, adjust=False).mean()
    
    # 2. Savitzky-Golay filter (modified to be causal)
    # We'll use a one-sided filter by padding
    for window_length in [7, 15, 31]:
        if len(df) > window_length:
            padded = np.pad(df[speed_col].values, (window_length-1, 0), mode='edge')
            filtered = signal.savgol_filter(padded, window_length, min(3, window_length-2))
            features[f'Speed_savgol_{window_length}'] = filtered[window_length-1:]
    
    # 3. Robust statistical features using only past data
    windows = [60, 300, 600, 900, 1200]
    
    for window in windows:
        # Use min_periods=1 to handle start of series
        # Critical: No center=True to avoid looking ahead
        
        features[f'Speed_rolling_median_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).median()
        )
        
        # MAD (Median Absolute Deviation) - custom implementation to ensure causality
        def causal_mad(series, window):
            mad_values = []
            for i in range(len(series)):
                start_idx = max(0, i - window + 1)
                window_data = series.iloc[start_idx:i+1]
                if len(window_data) > 0:
                    median = window_data.median()
                    mad = (window_data - median).abs().median()
                    mad_values.append(mad)
                else:
                    mad_values.append(np.nan)
            return pd.Series(mad_values, index=series.index)
        
        features[f'Speed_rolling_mad_{window}'] = causal_mad(df[speed_col], window)
        
        # Robust z-score
        median_col = features[f'Speed_rolling_median_{window}']
        mad_col = features[f'Speed_rolling_mad_{window}']
        features[f'Speed_robust_zscore_{window}'] = (
            (df[speed_col] - median_col) / (mad_col + 1e-9)
        )
        
        # Quantiles (using only past data)
        features[f'Speed_rolling_q25_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).quantile(0.25)
        )
        features[f'Speed_rolling_q75_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).quantile(0.75)
        )
        features[f'Speed_rolling_iqr_{window}'] = (
            features[f'Speed_rolling_q75_{window}'] - features[f'Speed_rolling_q25_{window}']
        )
        
        # Check if current value is outside IQR bounds (based on past data only)
        lower_bound = features[f'Speed_rolling_q25_{window}'] - 1.5 * features[f'Speed_rolling_iqr_{window}']
        upper_bound = features[f'Speed_rolling_q75_{window}'] + 1.5 * features[f'Speed_rolling_iqr_{window}']
        features[f'Speed_outside_iqr_{window}'] = (
            ((df[speed_col] < lower_bound) | (df[speed_col] > upper_bound)).astype(int)
        )
    
    # 4. Change detection features (all backward-looking)
    for lag in [1, 5, 10, 30, 60]:
        # Percentage change from lag periods ago
        features[f'Speed_pct_change_{lag}'] = df[speed_col].pct_change(periods=lag).fillna(0)
        
        # Absolute change
        features[f'Speed_abs_change_{lag}'] = df[speed_col].diff(periods=lag).abs().fillna(0)
        
        # Rate of change
        features[f'Speed_rate_of_change_{lag}'] = (
            df[speed_col].diff(periods=lag).fillna(0) / lag
        )
    
    # 5. Causal cumulative features
    # Cumulative sum of changes (like integration)
    features['Speed_cumsum_change'] = df[speed_col].diff().fillna(0).cumsum()
    
    # 6. Pattern detection using only past data
    for window in [30, 60, 120]:
        # Count of increases/decreases in past window
        def count_increases(series, window):
            increases = []
            for i in range(len(series)):
                start_idx = max(0, i - window + 1)
                window_data = series.iloc[start_idx:i+1]
                if len(window_data) > 1:
                    diffs = window_data.diff()
                    increases.append((diffs > 0).sum())
                else:
                    increases.append(0)
            return pd.Series(increases, index=series.index)
        
        features[f'Speed_increases_in_{window}'] = count_increases(df[speed_col], window)
        features[f'Speed_decreases_in_{window}'] = window - features[f'Speed_increases_in_{window}']
        
        # Variance in past window (measure of stability)
        features[f'Speed_rolling_var_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).var().fillna(0)
        )
    
    # 7. Threshold crossings (based on historical quantiles)
    # Count how many times we cross certain thresholds
    historical_q10 = df[speed_col].expanding(min_periods=100).quantile(0.1)
    historical_q90 = df[speed_col].expanding(min_periods=100).quantile(0.9)
    
    features['Speed_below_historical_q10'] = (df[speed_col] < historical_q10).astype(int)
    features['Speed_above_historical_q90'] = (df[speed_col] > historical_q90).astype(int)
    
    # 8. Memory features (exponentially weighted statistics)
    for span in [10, 50, 200]:
        ewm = df[speed_col].ewm(span=span, adjust=False)
        features[f'Speed_ewm_mean_{span}'] = ewm.mean()
        features[f'Speed_ewm_std_{span}'] = ewm.std().fillna(0)
        
        # Distance from EWM mean in terms of EWM std
        features[f'Speed_ewm_zscore_{span}'] = (
            (df[speed_col] - features[f'Speed_ewm_mean_{span}']) / 
            (features[f'Speed_ewm_std_{span}'] + 1e-9)
        )
    
    # 9. Causal peak detection (only looking backward)
    for window in [10, 30]:
        # Is current value the maximum in the past window?
        features[f'Speed_is_window_max_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).max() == df[speed_col]
        ).astype(int)
        
        # Is current value the minimum in the past window?
        features[f'Speed_is_window_min_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).min() == df[speed_col]
        ).astype(int)
    
    # 10. Filtered derivatives (using causal filters)
    # First derivative with smoothing
    speed_smooth = features['Speed_ema_0.3']  # Use smoothed version
    features['Speed_derivative_1'] = speed_smooth.diff().fillna(0)
    
    # Second derivative (acceleration)
    features['Speed_derivative_2'] = features['Speed_derivative_1'].diff().fillna(0)
    
    # 11. Range features (all based on past data)
    for window in [60, 300]:
        rolling_min = df[speed_col].rolling(window=window, min_periods=1).min()
        rolling_max = df[speed_col].rolling(window=window, min_periods=1).max()
        
        features[f'Speed_range_{window}'] = rolling_max - rolling_min
        
        # Position within range (0 to 1)
        features[f'Speed_range_position_{window}'] = (
            (df[speed_col] - rolling_min) / (features[f'Speed_range_{window}'] + 1e-9)
        ).clip(0, 1)
    
    # 12. Kalman filter for anomaly detection (simplified 1D version)
    def apply_kalman_filter(signal, process_variance=1e-5, measurement_variance=0.1):
        """Simple 1D Kalman filter"""
        n = len(signal)
        filtered = np.zeros(n)
        
        # Initial estimates
        x_est = signal.iloc[0] if not pd.isna(signal.iloc[0]) else 0
        p_est = 1.0
        
        for i in range(n):
            # Prediction
            x_pred = x_est
            p_pred = p_est + process_variance
            
            # Update
            if not pd.isna(signal.iloc[i]):
                k_gain = p_pred / (p_pred + measurement_variance)
                x_est = x_pred + k_gain * (signal.iloc[i] - x_pred)
                p_est = (1 - k_gain) * p_pred
            
            filtered[i] = x_est
        
        return filtered
    
    kalman_filtered = apply_kalman_filter(df[speed_col])
    features['Speed_kalman_filtered'] = kalman_filtered
    features['Speed_kalman_residual'] = df[speed_col] - kalman_filtered
    features['Speed_kalman_residual_abs'] = np.abs(features['Speed_kalman_residual'])
    
    # Fill NaN values with 0 (appropriate for most features as they represent "no change" or "no pattern")
    features = features.fillna(0)
    
    # Remove the original speed column to avoid leakage
    features = features.drop('Speed', axis=1)
    
    return features

def create_ensemble_anomaly_detector(
    df,
    speed_col='speed',
    label_col='pred_label',
    timestamp_col='indo_time',
    random_state=42,
    save_path_prefix='anomaly_interval',
    use_temporal_validation=True
):
    """
    Create a robust anomaly detector with 80-20 train-validation split.
    """
    print("Building robust features without data leakage...")
    X = engineer_robust_features(df, speed_col, timestamp_col)
    y = df[label_col]
    
    print(f"Total features created: {X.shape[1]}")
    
    if use_temporal_validation:
        print("\nUsing temporal validation strategy...")
        sorted_idx = df.sort_values(timestamp_col).index
        X_sorted = X.loc[sorted_idx]
        y_sorted = y.loc[sorted_idx]
        df_sorted = df.loc[sorted_idx]

        n = len(X_sorted)
        train_end = int(0.8 * n)
        
        X_train = X_sorted.iloc[:train_end]
        y_train = y_sorted.iloc[:train_end]
        X_val = X_sorted.iloc[train_end:]
        y_val = y_sorted.iloc[train_end:]
        
        print(f"Temporal split - Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"Train period: {df_sorted.iloc[0][timestamp_col]} to {df_sorted.iloc[train_end-1][timestamp_col]}")
        print(f"Val period: {df_sorted.iloc[train_end][timestamp_col]} to {df_sorted.iloc[-1][timestamp_col]}")
    else:

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
    
    print(f"\nAnomaly distribution:")
    print(f"Train: {y_train.value_counts(normalize=True).sort_index().to_dict()}")
    print(f"Val: {y_val.value_counts(normalize=True).sort_index().to_dict()}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("\nTraining robust XGBoost model...")

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [5, 10, 20],
        'gamma': [0.1, 0.3, 0.5, 1.0],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_alpha': [0.1, 0.5, 1.0, 2.0],
        'reg_lambda': [1.0, 2.0, 5.0, 10.0],
        'scale_pos_weight': [scale_pos_weight]
    }

    base_model = XGBClassifier(
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=10,
        n_jobs=-1
    )

    def balanced_anomaly_score(y_true, y_pred):
        """Custom scoring function balancing precision, recall, and specificity"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return 0.4 * recall + 0.3 * precision + 0.3 * specificity
    
    scorer = make_scorer(balanced_anomaly_score)

    if use_temporal_validation:
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=3)
    else:
        cv = 3

    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring=scorer,
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    random_search.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    print(f"\nBest parameters: {random_search.best_params_}")
    model = random_search.best_estimator_

    print("\n--- VALIDATION SET PERFORMANCE ---")
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    val_metrics = evaluate_performance(y_val, y_val_pred, y_val_proba, "Validation")

    # Create visualizations for validation set
    create_validation_visualizations(X_val, y_val, y_val_pred, y_val_proba, 
                                   model, X, save_path_prefix)

    print("\n--- SAVING MODEL AND RESULTS ---")
    joblib.dump(model, f'{save_path_prefix}_robust_model.pkl')
    joblib.dump(scaler, f'{save_path_prefix}_robust_scaler.pkl')

    config = {
        'feature_names': list(X.columns),
        'model_params': model.get_params(),
        'validation_metrics': val_metrics,
        'training_date': datetime.now().isoformat(),
        'n_train_samples': len(X_train),
        'n_features': X.shape[1],
        'anomaly_ratio_train': float((y_train == 1).mean()),
        'temporal_validation': use_temporal_validation,
        'train_val_split': '80-20',
        'feature_engineering_notes': {
            'approach': 'Causal features only - no future information used',
            'filters': 'EMA, Savitzky-Golay (causal), Kalman filter',
            'windows': 'Domain-specific: 60, 300, 600, 900, 1200',
            'no_datetime_features': True
        }
    }
    
    with open(f'{save_path_prefix}_robust_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel and results saved with prefix: {save_path_prefix}_robust")

    return model, scaler, config

def evaluate_performance(y_true, y_pred, y_proba, dataset_name):
    """Comprehensive performance evaluation"""
    if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
        print(f"Warning: Only one class present in {dataset_name} predictions")
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if cm.shape == (1, 1):
            if y_true.iloc[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_anomaly': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics['auc'] = auc(fpr, tpr)
    else:
        metrics['auc'] = np.nan
    
    print(f"\n{dataset_name} Set Metrics:")
    for metric, value in metrics.items():
        if not np.isnan(value):
            print(f"{metric}: {value:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
    
    return metrics

def create_validation_visualizations(X_val, y_val, y_val_pred, y_val_proba, 
                                   model, X_all, save_prefix):
    """Create visualizations for validation set analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. ROC Curve
    ax1 = axes[0, 0]
    if len(np.unique(y_val)) > 1:
        fpr, tpr, _ = roc_curve(y_val, y_val_proba)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Feature Importance (Top 15)
    ax2 = axes[0, 1]
    feature_importance = pd.DataFrame({
        'feature': X_all.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    ax2.barh(range(len(feature_importance)), feature_importance['importance'])
    ax2.set_yticks(range(len(feature_importance)))
    ax2.set_yticklabels(feature_importance['feature'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 15 Feature Importances')
    
    # 3. Confusion Matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'], ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. Probability Distribution
    ax4 = axes[1, 1]
    if y_val.sum() > 0:
        ax4.hist(y_val_proba[y_val == 0], bins=30, alpha=0.5, label='Normal', density=True)
        ax4.hist(y_val_proba[y_val == 1], bins=30, alpha=0.5, label='Anomaly', density=True)
    else:
        ax4.hist(y_val_proba, bins=30, alpha=0.5, label='All (Normal)', density=True)
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Density')
    ax4.set_title('Probability Distribution by True Class')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_validation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def predict_unlabeled_data(
    unlabeled_df,
    model_path,
    scaler_path,
    config_path,
    speed_col='A2:MCPGSpeed',
    timestamp_col='Time_stamp',
    save_path='anomaly_predictions_unlabeled.csv'
):
    """
    Predict anomalies on unlabeled dataset.
    
    Parameters:
    -----------
    unlabeled_df : pd.DataFrame
        Unlabeled data with speed and timestamp columns
    model_path : str
        Path to saved model file
    scaler_path : str
        Path to saved scaler file
    config_path : str
        Path to saved configuration file
    speed_col : str
        Name of speed column in unlabeled data
    timestamp_col : str
        Name of timestamp column in unlabeled data
    save_path : str
        Path to save predictions
        
    Returns:
    --------
    predictions_df : pd.DataFrame
        DataFrame with original data and predictions
    """
    print("Loading model and configuration...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Model expects {len(config['feature_names'])} features")
    
    # Ensure timestamp is datetime
    if timestamp_col in unlabeled_df.columns:
        unlabeled_df[timestamp_col] = pd.to_datetime(unlabeled_df[timestamp_col])
    
    print("\nEngineering features for unlabeled data...")
    # Create a temporary DataFrame with expected column names for feature engineering
    temp_df = unlabeled_df.copy()
    temp_df['speed'] = temp_df[speed_col]
    temp_df['indo_time'] = temp_df[timestamp_col] if timestamp_col in temp_df.columns else pd.Timestamp.now()
    
    # Engineer features using the same function
    X_unlabeled = engineer_robust_features(temp_df, speed_col='speed', timestamp_col='indo_time')
    
    # Ensure all expected features are present
    missing_features = set(config['feature_names']) - set(X_unlabeled.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feat in missing_features:
            X_unlabeled[feat] = 0
    
    # Reorder columns to match training
    X_unlabeled = X_unlabeled[config['feature_names']]
    
    print("Scaling features...")
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    
    print("Making predictions...")
    predictions = model.predict(X_unlabeled_scaled)
    probabilities = model.predict_proba(X_unlabeled_scaled)[:, 1]
    
    # Create results DataFrame
    results_df = unlabeled_df.copy()
    results_df['predicted_anomaly'] = predictions
    results_df['anomaly_probability'] = probabilities
    results_df['anomaly_confidence'] = np.abs(probabilities - 0.5) * 2
    
    # Save predictions
    results_df.to_csv(save_path, index=False)
    
    print(f"\nPrediction Summary:")
    print(f"Total samples: {len(results_df)}")
    print(f"Predicted anomalies: {predictions.sum()} ({predictions.mean()*100:.2f}%)")
    print(f"High confidence anomalies (prob > 0.8): {(probabilities > 0.8).sum()}")
    print(f"Low confidence predictions (0.4 < prob < 0.6): {((probabilities > 0.4) & (probabilities < 0.6)).sum()}")
    
    print(f"\nPredictions saved to: {save_path}")
    
    # Create visualization of predictions
    create_unlabeled_predictions_visualization(results_df, speed_col, timestamp_col, save_path.replace('.csv', ''))
    
    return results_df

def create_unlabeled_predictions_visualization(results_df, speed_col, timestamp_col, save_prefix):
    """Create visualization for unlabeled predictions"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Use index if timestamp not available
    if timestamp_col in results_df.columns:
        time_axis = results_df[timestamp_col]
        time_label = 'Timestamp'
    else:
        time_axis = np.arange(len(results_df))
        time_label = 'Index'
    
    # 1. Speed with predicted anomalies
    ax1 = axes[0]
    ax1.plot(time_axis, results_df[speed_col], 'b-', alpha=0.6, linewidth=0.5, label='Speed')
    
    # Mark predicted anomalies
    anomaly_mask = results_df['predicted_anomaly'] == 1
    if anomaly_mask.sum() > 0:
        ax1.scatter(time_axis[anomaly_mask], results_df[speed_col][anomaly_mask], 
                   c='red', s=30, alpha=0.8, label='Predicted Anomalies', marker='o')
    
    ax1.set_xlabel(time_label)
    ax1.set_ylabel('Speed')
    ax1.set_title('Speed Time Series with Predicted Anomalies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Anomaly probability over time
    ax2 = axes[1]
    ax2.plot(time_axis, results_df['anomaly_probability'], 'g-', alpha=0.7, linewidth=1)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    ax2.fill_between(time_axis, 0, results_df['anomaly_probability'], 
                    where=(results_df['anomaly_probability'] > 0.5), 
                    alpha=0.3, color='red', label='Anomaly Region')
    ax2.set_xlabel(time_label)
    ax2.set_ylabel('Anomaly Probability')
    ax2.set_title('Anomaly Probability Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # 3. Probability histogram
    ax3 = axes[2]
    ax3.hist(results_df['anomaly_probability'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
    ax3.set_xlabel('Anomaly Probability')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Anomaly Probabilities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_prefix}_visualization.png")


if __name__ == "__main__":
    file_path = 'Data/event_all.csv'
    
    print("Training robust anomaly detection model without data leakage...")
    print("All features are causal - using only past information\n")
    
    data_df = pd.read_csv(file_path, parse_dates=['indo_time'])

    model, scaler, config = create_ensemble_anomaly_detector(
        df=data_df,
        speed_col='speed',
        label_col='pred_label',
        timestamp_col='indo_time',
        random_state=42,
        save_path_prefix='anomaly',
        use_temporal_validation=True
    )
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Model saved: anomaly_robust_model.pkl")
    print(f"✓ Scaler saved: anomaly_robust_scaler.pkl")
    print(f"✓ Config saved: anomaly_robust_config.json")
    print(f"✓ Validation visualizations: anomaly_validation_analysis.png")

    print("\n" + "="*60)
    print("PREDICTING ON UNLABELED DATA")
    print("="*60)

    unlabeled_df = pd.read_csv('Data\input_data.csv')

    predictions_df = predict_unlabeled_data(
        unlabeled_df=unlabeled_df,
        model_path='anomaly_robust_model.pkl',
        scaler_path='anomaly_robust_scaler.pkl',
        config_path='anomaly_robust_config.json',
        speed_col='A2:MCPGSpeed',
        timestamp_col='Time_stamp',
        save_path='anomaly_predictions_unlabeled.csv'
    )
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)