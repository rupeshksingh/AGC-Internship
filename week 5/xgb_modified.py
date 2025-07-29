import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, make_scorer
)
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import joblib
import json
from datetime import datetime
warnings.filterwarnings('ignore')

def detect_data_drift(train_data, new_data, features, threshold=0.1):
    """
    Detect if there's significant drift between training and new data.
    """
    drift_report = {}
    significant_drifts = []
    
    for feature in features:
        if feature in train_data.columns and feature in new_data.columns:
            # Calculate KS statistic
            ks_stat, p_value = stats.ks_2samp(train_data[feature], new_data[feature])
            
            # Calculate distribution statistics
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

def create_temporal_validation_splits(X, y, df, timestamp_col='indo_time', n_splits=3):
    """
    Create time-based validation splits to better simulate real-world deployment.
    """
    sorted_indices = df.sort_values(timestamp_col).index
    X_sorted = X.loc[sorted_indices]
    y_sorted = y.loc[sorted_indices]

    splits = []
    total_size = len(X_sorted)
    
    for i in range(n_splits):
        val_start = int(total_size * (0.6 + i * 0.1))
        val_end = int(total_size * (0.7 + i * 0.1))
        
        train_idx = sorted_indices[:val_start]
        val_idx = sorted_indices[val_start:val_end]
        
        splits.append((train_idx, val_idx))
    
    return splits

def engineer_robust_features(df, speed_col='speed', timestamp_col='indo_time'):
    """
    Enhanced feature engineering with more robust features for generalization.
    """
    features = pd.DataFrame(df[speed_col]).copy()
    features.columns = ['Speed']

    original_index = df.index
    
    # 1. Robust statistical features (less sensitive to distribution shifts)
    windows = [60, 300, 600, 900, 1200]
    
    for window in windows:
        min_periods = max(1, window // 50)
        
        # Robust statistics using median and MAD
        features[f'Speed_rolling_median_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=min_periods).median()
        )
        
        # Median Absolute Deviation (more robust than std)
        rolling_median = features[f'Speed_rolling_median_{window}']
        features[f'Speed_rolling_mad_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=min_periods)
            .apply(lambda x: np.median(np.abs(x - np.median(x))))
        )
        
        # Robust z-score using median and MAD
        features[f'Speed_robust_zscore_{window}'] = (
            (df[speed_col] - rolling_median) / 
            (features[f'Speed_rolling_mad_{window}'] + 1e-9)
        )
        
        # Quantile-based features (more stable)
        features[f'Speed_rolling_q25_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=min_periods).quantile(0.25)
        )
        features[f'Speed_rolling_q75_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=min_periods).quantile(0.75)
        )
        features[f'Speed_rolling_iqr_{window}'] = (
            features[f'Speed_rolling_q75_{window}'] - features[f'Speed_rolling_q25_{window}']
        )
        
        # Percentage within IQR (distribution-free)
        lower_bound = features[f'Speed_rolling_q25_{window}'] - 1.5 * features[f'Speed_rolling_iqr_{window}']
        upper_bound = features[f'Speed_rolling_q75_{window}'] + 1.5 * features[f'Speed_rolling_iqr_{window}']
        features[f'Speed_outside_iqr_{window}'] = (
            ((df[speed_col] < lower_bound) | (df[speed_col] > upper_bound)).astype(int)
        )
    
    # 2. Relative features (normalized by recent context)
    for window in [60, 300]:
        # Percentage deviation from median
        median_col = f'Speed_rolling_median_{window}'
        features[f'Speed_pct_dev_from_median_{window}'] = (
            (df[speed_col] - features[median_col]) / (features[median_col] + 1e-9)
        )
        
        # Rank within window (distribution-free)
        features[f'Speed_rank_in_window_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1)
            .apply(lambda x: stats.rankdata(x)[-1] / len(x))
        )
    
    # 3. Change detection features
    for lag in [1, 5, 10, 30, 60]:
        # Relative changes instead of absolute
        features[f'Speed_pct_change_{lag}'] = df[speed_col].pct_change(periods=lag).fillna(0)
        
        # Smoothed differences
        features[f'Speed_smooth_diff_{lag}'] = (
            df[speed_col].rolling(window=5, min_periods=1).mean().diff(periods=lag).fillna(0)
        )
    
    # 4. Pattern-based features
    # Local trend strength
    for window in [30, 60]:
        def trend_strength(x):
            if len(x) < 3:
                return 0
            t = np.arange(len(x))
            slope, _, r_value, _, _ = stats.linregress(t, x)
            return r_value ** 2  # R-squared as trend strength
        
        features[f'Speed_trend_strength_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=3)
            .apply(trend_strength, raw=True).fillna(0)
        )
    
    # 5. Stability features
    # Count of direction changes
    for window in [30, 60]:
        def count_direction_changes(x):
            if len(x) < 3:
                return 0
            diff = np.diff(x)
            return np.sum(diff[:-1] * diff[1:] < 0)
        
        features[f'Speed_direction_changes_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=3)
            .apply(count_direction_changes, raw=True).fillna(0)
        )
    
    # 6. Adaptive threshold features
    # Dynamic bounds based on recent variability
    for window in [60, 300]:
        center = features[f'Speed_rolling_median_{window}']
        scale = features[f'Speed_rolling_mad_{window}']
        
        # How many MADs away from median
        features[f'Speed_n_mads_from_median_{window}'] = (
            np.abs(df[speed_col] - center) / (scale + 1e-9)
        )
    
    # 7. Ensemble of simple anomaly detectors
    # Simple outlier flags that can be combined
    features['Speed_is_local_min'] = (
        (df[speed_col] < df[speed_col].shift(1)) & 
        (df[speed_col] < df[speed_col].shift(-1))
    ).astype(int)
    
    features['Speed_is_local_max'] = (
        (df[speed_col] > df[speed_col].shift(1)) & 
        (df[speed_col] > df[speed_col].shift(-1))
    ).astype(int)

    if timestamp_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        features['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
        features['minute'] = pd.to_datetime(df[timestamp_col]).dt.minute
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

        time_diff = (pd.to_datetime(df[timestamp_col]) - pd.to_datetime(df[timestamp_col]).min()).dt.total_seconds()
        features['time_position'] = time_diff / (time_diff.max() + 1e-9)

    features = features.fillna(0)

    features.index = original_index
    
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
    Create a more robust anomaly detector using ensemble approach and temporal validation.
    """
    print("Building robust features...")
    X = engineer_robust_features(df, speed_col, timestamp_col)
    y = df[label_col]
    
    print(f"Total features created: {X.shape[1]}")
    
    if use_temporal_validation:
        print("\nUsing temporal validation strategy...")
        sorted_idx = df.sort_values(timestamp_col).index
        X_sorted = X.loc[sorted_idx]
        y_sorted = y.loc[sorted_idx]

        n = len(X_sorted)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        X_train = X_sorted.iloc[:train_end]
        y_train = y_sorted.iloc[:train_end]
        X_val = X_sorted.iloc[train_end:val_end]
        y_val = y_sorted.iloc[train_end:val_end]
        X_test = X_sorted.iloc[val_end:]
        y_test = y_sorted.iloc[val_end:]
        
        print(f"Temporal split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Train period: {df.loc[X_train.index[0], timestamp_col]} to {df.loc[X_train.index[-1], timestamp_col]}")
        print(f"Val period: {df.loc[X_val.index[0], timestamp_col]} to {df.loc[X_val.index[-1], timestamp_col]}")
        print(f"Test period: {df.loc[X_test.index[0], timestamp_col]} to {df.loc[X_test.index[-1], timestamp_col]}")
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=random_state
        )
    
    print(f"\nAnomaly distribution:")
    print(f"Train: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Val: {y_val.value_counts(normalize=True).to_dict()}")
    print(f"Test: {y_test.value_counts(normalize=True).to_dict()}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("\nTraining robust XGBoost model...")

    param_grid = {
        'n_estimators': [50, 100, 150],  # Fewer trees to avoid overfitting
        'max_depth': [3, 4, 5, 6],  # Shallower trees
        'learning_rate': [0.01, 0.05, 0.1],  # Lower learning rates
        'min_child_weight': [5, 10, 20],  # Higher values for more conservative splits
        'gamma': [0.1, 0.3, 0.5, 1.0],  # Higher regularization
        'subsample': [0.6, 0.7, 0.8],  # More conservative subsampling
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_alpha': [0.1, 0.5, 1.0, 2.0],  # L1 regularization
        'reg_lambda': [1.0, 2.0, 5.0, 10.0],  # L2 regularization
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
        n_iter=30,
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
    
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
        'f1_anomaly': f1_score(y_val, y_val_pred, pos_label=1),
        'custom_score': balanced_anomaly_score(y_val, y_val_pred)
    }
    
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n--- TEST SET PERFORMANCE ---")
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'f1_anomaly': f1_score(y_test, y_test_pred, pos_label=1),
        'custom_score': balanced_anomaly_score(y_test, y_test_pred)
    }
    
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n--- PERFORMANCE ANALYSIS ---")
    for metric in ['accuracy', 'balanced_accuracy', 'f1_anomaly']:
        degradation = (val_metrics[metric] - test_metrics[metric]) / val_metrics[metric] * 100
        print(f"{metric} degradation: {degradation:.2f}%")

    print("\n--- DATA DRIFT ANALYSIS ---")
    drift_report, significant_drifts = detect_data_drift(
        X_val, X_test, X.columns[:20]
    )
    
    if significant_drifts:
        print(f"Significant drift detected in {len(significant_drifts)} features:")
        for feat in significant_drifts[:5]:
            print(f"  - {feat}: KS={drift_report[feat]['ks_statistic']:.3f}")
    else:
        print("No significant data drift detected")

    print("\n--- SAVING MODEL ---")

    joblib.dump(model, f'{save_path_prefix}_robust_model.pkl')
    joblib.dump(scaler, f'{save_path_prefix}_robust_scaler.pkl')

    config = {
        'feature_names': list(X.columns),
        'model_params': model.get_params(),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_date': datetime.now().isoformat(),
        'n_train_samples': len(X_train),
        'n_features': X.shape[1],
        'anomaly_ratio_train': float((y_train == 1).mean()),
        'temporal_validation': use_temporal_validation,
        'performance_thresholds': {
            'min_f1_anomaly': 0.5,
            'max_degradation': 0.15
        }
    }
    
    with open(f'{save_path_prefix}_robust_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved with prefix: {save_path_prefix}_robust")

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    val_calibration = []
    for i in range(n_bins):
        mask = (y_val_proba >= bin_edges[i]) & (y_val_proba < bin_edges[i+1])
        if mask.sum() > 0:
            val_calibration.append(y_val[mask].mean())
        else:
            val_calibration.append(np.nan)

    test_calibration = []
    for i in range(n_bins):
        mask = (y_test_proba >= bin_edges[i]) & (y_test_proba < bin_edges[i+1])
        if mask.sum() > 0:
            test_calibration.append(y_test[mask].mean())
        else:
            test_calibration.append(np.nan)
    
    plt.plot(bin_centers, val_calibration, 'o-', label='Validation', linewidth=2)
    plt.plot(bin_centers, test_calibration, 's-', label='Test', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')

    plt.subplot(2, 2, 3)
    plt.hist(y_val_proba[y_val == 0], bins=30, alpha=0.5, label='Normal (Val)', density=True)
    plt.hist(y_val_proba[y_val == 1], bins=30, alpha=0.5, label='Anomaly (Val)', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if use_temporal_validation:
        plt.subplot(2, 2, 4)
        window = 1000
        rolling_predictions = []
        rolling_timestamps = []
        
        test_sorted_idx = X_test.index
        for i in range(window, len(y_test_pred), 100):
            y_true_window = y_test.iloc[i-window:i]
            y_pred_window = y_test_pred[i-window:i]
            if len(np.unique(y_true_window)) > 1:
                score = balanced_anomaly_score(y_true_window, y_pred_window)
                rolling_predictions.append(score)
                rolling_timestamps.append(i)
        
        if rolling_predictions:
            plt.plot(rolling_timestamps, rolling_predictions)
            plt.axhline(y=np.mean(rolling_predictions), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(rolling_predictions):.3f}')
            plt.xlabel('Sample Index')
            plt.ylabel('Performance Score')
            plt.title('Rolling Performance on Test Set')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_robust_analysis.png', dpi=150)
    plt.close()
    
    return model, scaler, config

def predict_with_monitoring(
    new_data_path,
    model_path,
    scaler_path,
    config_path,
    output_path,
    speed_col='A2:MCPGSpeed',
    timestamp_col='Time_stamp',
    check_drift=True
):
    """
    Predict on new data with performance monitoring and drift detection.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    new_df = pd.read_csv(new_data_path, parse_dates=[timestamp_col])
    new_df = new_df.rename(columns={speed_col: 'speed', timestamp_col: 'indo_time'})

    X_new = engineer_robust_features(new_df, 'speed', 'indo_time')
    X_new = X_new[config['feature_names']]
    X_new_scaled = scaler.transform(X_new)

    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]

    anomaly_rate = predictions.mean()
    expected_rate = config['anomaly_ratio_train']
    rate_deviation = abs(anomaly_rate - expected_rate) / expected_rate

    warnings = []
    
    if rate_deviation > 0.5:
        warnings.append(f"Anomaly rate ({anomaly_rate:.3f}) significantly different from training ({expected_rate:.3f})")
    
    if probabilities.max() < 0.8:
        warnings.append("No high-confidence anomalies detected")

    if check_drift:
        feature_stats = pd.DataFrame(X_new_scaled).describe()
        extreme_features = []
        
        for col_idx, col_name in enumerate(config['feature_names'][:20]):
            col_values = X_new_scaled[:, col_idx]
            if np.abs(col_values).max() > 5:
                extreme_features.append(col_name)
        
        if extreme_features:
            warnings.append(f"Extreme values detected in features: {', '.join(extreme_features[:5])}")

    results_df = pd.DataFrame({
        'timestamp': new_df['indo_time'],
        'speed': new_df['speed'],
        'anomaly_probability': probabilities,
        'predicted_label': predictions,
        'anomaly_status': ['Anomaly' if p == 1 else 'Normal' for p in predictions]
    })

    metadata = {
        'prediction_date': datetime.now().isoformat(),
        'n_samples': len(results_df),
        'anomaly_rate': float(anomaly_rate),
        'expected_anomaly_rate': expected_rate,
        'max_probability': float(probabilities.max()),
        'warnings': warnings,
        'performance_check': 'PASS' if len(warnings) == 0 else 'WARNING'
    }

    results_df.to_csv(output_path, index=False)

    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("PREDICTION SUMMARY WITH MONITORING")
    print("="*60)
    print(f"Samples processed: {len(results_df)}")
    print(f"Anomalies detected: {predictions.sum()} ({anomaly_rate:.2%})")
    print(f"Expected anomaly rate: {expected_rate:.2%}")
    print(f"Status: {metadata['performance_check']}")
    
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    
    return results_df, metadata


if __name__ == "__main__":
    file_path = 'Data\event_all.csv'
    
    print("Training robust anomaly detection model...")
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
    
    print("\nModel training complete!")

    results, metadata = predict_with_monitoring(
        new_data_path='Data\input_data.csv',
        model_path='anomaly_robust_model.pkl',
        scaler_path='anomaly_robust_scaler.pkl',
        config_path='anomaly_robust_config.json',
        output_path='predictions.csv',
        check_drift=True
    )