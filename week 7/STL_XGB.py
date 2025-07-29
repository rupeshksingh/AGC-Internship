import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, make_scorer
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import joblib
import json
from datetime import datetime
warnings.filterwarnings('ignore')


class CausalSTLDecomposer:
    """
    STL decomposition that respects causality for production use.
    Uses sliding windows to avoid future data leakage.
    """
    def __init__(self, 
                 seasonal_periods=[60, 300, 900],
                 trend_window=901,
                 seasonal_window=61,
                 robust=True):
        """
        Args:
            seasonal_periods: List of seasonal periods to extract
            trend_window: Window size for trend extraction (must be odd)
            seasonal_window: Window for seasonal smoothing
            robust: Use robust STL (resistant to outliers)
        """
        self.seasonal_periods = seasonal_periods
        self.trend_window = trend_window if trend_window % 2 == 1 else trend_window + 1
        self.seasonal_window = seasonal_window if seasonal_window % 2 == 1 else seasonal_window + 1
        self.robust = robust
        self.decomposers = {}
        
    def fit_transform_windowed(self, data, window_size=7200):
        """
        Apply STL in sliding windows to respect causality.
        For production, we can only use past data.
        """
        n = len(data)
        results = {
            'trend': np.full(n, np.nan),
            'seasonal': np.full(n, np.nan),
            'residual': np.full(n, np.nan),
            'strength_trend': np.full(n, np.nan),
            'strength_seasonal': np.full(n, np.nan)
        }

        for period in self.seasonal_periods:
            results[f'seasonal_{period}'] = np.full(n, np.nan)
            results[f'residual_{period}'] = np.full(n, np.nan)

        min_window = max(self.seasonal_periods) * 2
        
        for end_idx in range(min_window, n + 1):
            start_idx = max(0, end_idx - window_size)
            window_data = data[start_idx:end_idx]

            if len(window_data) < min_window:
                continue

            primary_period = max(self.seasonal_periods)
            try:
                stl = STL(
                    endog=window_data,
                    period=primary_period,
                    robust=self.robust
                )
                decomposition = stl.fit()

                current_idx = end_idx - 1
                results['trend'][current_idx] = decomposition.trend[-1]
                results['seasonal'][current_idx] = decomposition.seasonal[-1]
                results['residual'][current_idx] = decomposition.resid[-1]

                var_resid = np.var(decomposition.resid)

                detrended = window_data - decomposition.seasonal
                results['strength_trend'][current_idx] = max(0, 1 - var_resid / np.var(detrended))

                deseasonalized = window_data - decomposition.trend
                results['strength_seasonal'][current_idx] = max(0, 1 - var_resid / np.var(deseasonalized))
                
            except Exception as e:
                results['trend'][current_idx] = np.mean(window_data[-self.trend_window:])
                results['residual'][current_idx] = window_data[-1] - results['trend'][current_idx]
                results['seasonal'][current_idx] = 0

            for period in self.seasonal_periods:
                if period >= len(window_data):
                    continue
                    
                try:
                    stl_multi = STL(
                        window_data,
                        period=period,
                        robust=self.robust
                    )
                    decomp_multi = stl_multi.fit()
                    
                    results[f'seasonal_{period}'][current_idx] = decomp_multi.seasonal[-1]
                    results[f'residual_{period}'][current_idx] = decomp_multi.resid[-1]
                    
                except:
                    seasonal_avg = np.mean([
                        window_data[i] for i in range(len(window_data)) 
                        if (len(window_data) - 1 - i) % period == 0
                    ])
                    results[f'seasonal_{period}'][current_idx] = seasonal_avg
                    results[f'residual_{period}'][current_idx] = window_data[-1] - seasonal_avg
        
        return results
    
    def transform_realtime(self, speed_buffer):
        """
        Transform data in real-time using only past information.
        """
        if len(speed_buffer) < max(self.seasonal_periods) * 2:
            return None

        window_size = min(7200, len(speed_buffer))
        data = np.array(list(speed_buffer)[-window_size:])

        results = self.fit_transform_windowed(data, window_size)

        return {k: v[-1] for k, v in results.items() if not np.isnan(v[-1])}


def engineer_stl_features(stl_results, speed_data, window_sizes=[30, 60, 150, 300, 600]):
    """
    Engineer features from STL decomposition results.
    Focuses on residual patterns that indicate anomalies.
    """
    n = len(speed_data)
    features = pd.DataFrame(index=range(n))

    features['speed'] = speed_data
    features['stl_trend'] = stl_results['trend']
    features['stl_seasonal'] = stl_results['seasonal']
    features['stl_residual'] = stl_results['residual']
    features['stl_strength_trend'] = stl_results['strength_trend']
    features['stl_strength_seasonal'] = stl_results['strength_seasonal']

    for period in [60, 300, 3600]:
        if f'residual_{period}' in stl_results:
            features[f'stl_residual_{period}'] = stl_results[f'residual_{period}']

    for window in window_sizes:
        features[f'residual_mean_{window}'] = (
            pd.Series(stl_results['residual']).rolling(window, min_periods=1).mean()
        )
        features[f'residual_std_{window}'] = (
            pd.Series(stl_results['residual']).rolling(window, min_periods=1).std()
        )
        features[f'residual_abs_mean_{window}'] = (
            pd.Series(np.abs(stl_results['residual'])).rolling(window, min_periods=1).mean()
        )

        features[f'residual_median_{window}'] = (
            pd.Series(stl_results['residual']).rolling(window, min_periods=1).median()
        )
        features[f'residual_mad_{window}'] = (
            pd.Series(stl_results['residual']).rolling(window, min_periods=1)
            .apply(lambda x: np.median(np.abs(x - np.median(x))))
        )

        rolling_median = features[f'residual_median_{window}']
        rolling_mad = features[f'residual_mad_{window}']
        features[f'residual_zscore_mad_{window}'] = (
            (stl_results['residual'] - rolling_median) / (rolling_mad + 1e-8)
        )

        features[f'residual_exceeds_3mad_{window}'] = (
            np.abs(features[f'residual_zscore_mad_{window}']) > 3
        ).astype(int)

    for lag in [1, 5, 10, 30, 60]:
        features[f'residual_diff_{lag}'] = pd.Series(stl_results['residual']).diff(lag)
        features[f'residual_abs_diff_{lag}'] = np.abs(features[f'residual_diff_{lag}'])

    features['speed_minus_trend'] = speed_data - stl_results['trend']
    features['speed_trend_ratio'] = speed_data / (stl_results['trend'] + 1e-8)

    features['trend_diff_10'] = pd.Series(stl_results['trend']).diff(10)
    features['trend_acceleration'] = pd.Series(stl_results['trend']).diff().diff()

    features['speed_minus_seasonal'] = speed_data - stl_results['seasonal']
    features['deseasonalized'] = speed_data - stl_results['seasonal']

    for period in [60, 300]:
        if f'seasonal_{period}' in stl_results:
            features[f'deviation_seasonal_{period}'] = (
                speed_data - stl_results[f'seasonal_{period}']
            )

    features['trend_plus_seasonal'] = stl_results['trend'] + stl_results['seasonal']
    features['actual_vs_expected'] = speed_data - features['trend_plus_seasonal']
    features['actual_vs_expected_ratio'] = speed_data / (features['trend_plus_seasonal'] + 1e-8)

    high_residual = (np.abs(stl_results['residual']) > 
                    2 * pd.Series(stl_results['residual']).rolling(300, min_periods=30).std())
    features['consecutive_high_residuals'] = (
        high_residual.groupby((high_residual != high_residual.shift()).cumsum()).cumsum()
    )

    if all(f'residual_{p}' in stl_results for p in [60, 300]):
        features['multi_scale_residual_agreement'] = (
            (np.abs(stl_results['residual_60']) > np.nanstd(stl_results['residual_60']) * 2) &
            (np.abs(stl_results['residual_300']) > np.nanstd(stl_results['residual_300']) * 2)
        ).astype(int)

    for window in [60, 300]:
        features[f'residual_skew_{window}'] = (
            pd.Series(stl_results['residual']).rolling(window, min_periods=30)
            .apply(lambda x: stats.skew(x))
        )
        features[f'residual_kurtosis_{window}'] = (
            pd.Series(stl_results['residual']).rolling(window, min_periods=30)
            .apply(lambda x: stats.kurtosis(x))
        )

    features['low_component_strength'] = (
        (stl_results['strength_trend'] < 0.3) | 
        (stl_results['strength_seasonal'] < 0.3)
    ).astype(int)

    features['anomaly_score_simple'] = (
        np.abs(features['residual_zscore_mad_60']) * 
        (2 - stl_results['strength_trend'] - stl_results['strength_seasonal'])
    )

    features = features.fillna(method='ffill').fillna(0)
    
    return features


def create_stl_xgb_detector(
    df,
    speed_col='speed',
    label_col='pred_label',
    timestamp_col='indo_time',
    random_state=42,
    save_path_prefix='stl_xgb_anomaly'
):
    """
    Create STL-XGBoost anomaly detector.
    """
    print("="*60)
    print("TRAINING STL-XGBOOST ANOMALY DETECTOR")
    print("="*60)

    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    print(f"\nData split (temporal):")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values
    
    print(f"\nAnomaly rates:")
    print(f"  Train: {np.mean(y_train == 1):.3%}")
    print(f"  Val: {np.mean(y_val == 1):.3%}")
    print(f"  Test: {np.mean(y_test == 1):.3%}")

    print("\n--- Step 1: STL Decomposition ---")
    
    stl_decomposer = CausalSTLDecomposer(
        robust=True
    )

    print("Decomposing training data...")
    train_stl = stl_decomposer.fit_transform_windowed(
        train_df[speed_col].values, 
        window_size=7200
    )
    
    print("Decomposing validation data...")
    val_stl = stl_decomposer.fit_transform_windowed(
        val_df[speed_col].values,
        window_size=7200
    )
    
    print("Decomposing test data...")
    test_stl = stl_decomposer.fit_transform_windowed(
        test_df[speed_col].values,
        window_size=7200
    )

    print("\n--- Step 2: Engineering Features ---")
    
    window_sizes = [30, 60, 150, 300, 600]
    
    X_train = engineer_stl_features(train_stl, train_df[speed_col].values, window_sizes)
    X_val = engineer_stl_features(val_stl, val_df[speed_col].values, window_sizes)
    X_test = engineer_stl_features(test_stl, test_df[speed_col].values, window_sizes)
    
    print(f"Feature dimensions: {X_train.shape[1]}")

    for X, df_temp in [(X_train, train_df), (X_val, val_df), (X_test, test_df)]:
        X['hour'] = pd.to_datetime(df_temp[timestamp_col]).dt.hour
        X['minute'] = pd.to_datetime(df_temp[timestamp_col]).dt.minute
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['minute_sin'] = np.sin(2 * np.pi * X['minute'] / 60)
        X['minute_cos'] = np.cos(2 * np.pi * X['minute'] / 60)

    print("\n--- Step 3: Handling Missing Values ---")

    min_samples = max(window_sizes)

    X_train = X_train.iloc[min_samples:].reset_index(drop=True)
    y_train = y_train[min_samples:]
    
    X_val = X_val.iloc[min_samples:].reset_index(drop=True)
    y_val = y_val[min_samples:]
    
    X_test = X_test.iloc[min_samples:].reset_index(drop=True)
    y_test = y_test[min_samples:]
    
    print(f"After trimming warm-up period:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    print("\n--- Step 4: Scaling Features ---")
    
    scaler = RobustScaler()

    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_val_clean = X_val.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)

    train_medians = X_train_clean.median()
    X_train_clean = X_train_clean.fillna(train_medians)
    X_val_clean = X_val_clean.fillna(train_medians)
    X_test_clean = X_test_clean.fillna(train_medians)

    for col in X_train_clean.columns:
        p999 = X_train_clean[col].quantile(0.999)
        p001 = X_train_clean[col].quantile(0.001)
        X_train_clean[col] = X_train_clean[col].clip(p001, p999)
        X_val_clean[col] = X_val_clean[col].clip(p001, p999)
        X_test_clean[col] = X_test_clean[col].clip(p001, p999)

    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    feature_names = list(X_train.columns)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    print("\n--- Step 5: Training XGBoost ---")

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [5, 10, 20],
        'gamma': [0.1, 0.3, 0.5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [1.0, 2.0, 5.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    base_model = XGBClassifier(
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=10,
        n_jobs=4
    )

    def balanced_anomaly_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0.4 * recall + 0.3 * precision + 0.3 * specificity
    
    scorer = make_scorer(balanced_anomaly_score)

    from sklearn.model_selection import TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=3)

    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring=scorer,
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=4
    )
    
    random_search.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    print(f"\nBest parameters: {random_search.best_params_}")
    xgb_model = random_search.best_estimator_

    print("\n--- VALIDATION SET PERFORMANCE ---")
    y_val_pred = xgb_model.predict(X_val_scaled)
    y_val_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, pos_label=1)
    
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Balanced Accuracy: {val_balanced_acc:.4f}")
    print(f"F1 Score (Anomaly): {val_f1:.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    print(f"Precision: {tp/(tp+fp):.3f}, Recall: {tp/(tp+fn):.3f}")
    
    print("\n--- TEST SET PERFORMANCE ---")
    y_test_pred = xgb_model.predict(X_test_scaled)
    y_test_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
    
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"F1 Score (Anomaly): {test_f1:.4f}")

    print("\n--- Top 20 Important Features ---")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20))

    print("\n--- Saving Models ---")

    stl_config = {
        'seasonal_periods': stl_decomposer.seasonal_periods,
        'trend_window': stl_decomposer.trend_window,
        'seasonal_window': stl_decomposer.seasonal_window,
        'robust': stl_decomposer.robust,
        'window_sizes': window_sizes
    }
    
    joblib.dump(stl_config, f'{save_path_prefix}_stl_config.pkl')
    joblib.dump(xgb_model, f'{save_path_prefix}_xgb_model.pkl')
    joblib.dump(scaler, f'{save_path_prefix}_scaler.pkl')

    config = {
        'feature_names': feature_names,
        'train_medians': train_medians.to_dict(),
        'train_percentiles': {
            col: {
                'p001': float(X_train_clean[col].quantile(0.001)),
                'p999': float(X_train_clean[col].quantile(0.999))
            } for col in X_train_clean.columns
        },
        'stl_config': stl_config,
        'min_samples': min_samples,
        'xgb_params': xgb_model.get_params(),
        'val_metrics': {
            'accuracy': val_accuracy,
            'balanced_accuracy': val_balanced_acc,
            'f1_score': val_f1
        },
        'test_metrics': {
            'accuracy': test_accuracy,
            'balanced_accuracy': test_balanced_acc,
            'f1_score': test_f1
        },
        'training_date': datetime.now().isoformat(),
        'n_train_samples': len(X_train),
        'anomaly_ratio_train': float(np.mean(y_train == 1))
    }
    
    with open(f'{save_path_prefix}_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModels saved with prefix: {save_path_prefix}")

    visualize_stl_results(test_df, test_stl, y_test, y_test_pred, 
                         y_test_proba, save_path_prefix)
    
    return stl_decomposer, xgb_model, scaler, config


def visualize_stl_results(test_df, stl_results, y_true, y_pred, y_proba, save_prefix):
    """Visualize STL decomposition and detection results."""
    fig, axes = plt.subplots(5, 1, figsize=(15, 15))

    speed = test_df['speed'].values
    time_index = test_df.index

    ax1 = axes[0]
    normal_mask = y_pred == 0
    anomaly_mask = y_pred == 1
    
    ax1.plot(time_index, speed, 'b-', alpha=0.5, label='Speed')
    ax1.scatter(time_index[anomaly_mask], speed[anomaly_mask],
               c='red', s=20, alpha=0.8, label='Detected Anomaly')

    true_anomaly_mask = y_true == 1
    ax1.scatter(time_index[true_anomaly_mask], speed[true_anomaly_mask],
               facecolors='none', edgecolors='green', s=30, alpha=0.7,
               label='True Anomaly', linewidths=1.5)
    
    ax1.set_ylabel('Speed')
    ax1.set_title('Original Speed with Anomalies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(time_index, stl_results['trend'], 'g-', label='Trend', linewidth=2)
    ax2.plot(time_index, speed, 'b-', alpha=0.3, label='Original')
    ax2.set_ylabel('Value')
    ax2.set_title('Trend Component')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(time_index, stl_results['seasonal'], 'orange', label='Seasonal')
    ax3.set_ylabel('Value')
    ax3.set_title('Seasonal Component')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4.plot(time_index, stl_results['residual'], 'gray', alpha=0.5, label='Residual')

    residual_std = np.nanstd(stl_results['residual'])
    high_residual_mask = np.abs(stl_results['residual']) > 2 * residual_std
    
    ax4.scatter(time_index[high_residual_mask & anomaly_mask], 
               stl_results['residual'][high_residual_mask & anomaly_mask],
               c='red', s=20, alpha=0.8, label='Anomaly (High Residual)')
    
    ax4.axhline(y=2*residual_std, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(y=-2*residual_std, color='r', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Residual')
    ax4.set_title('Residual Component (±2σ threshold shown)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = axes[4]
    ax5.plot(time_index, y_proba, 'purple', linewidth=1)
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax5.fill_between(time_index, y_proba, 0.5,
                    where=(y_proba > 0.5), color='red', alpha=0.3)
    ax5.set_ylabel('Probability')
    ax5.set_xlabel('Sample Index')
    ax5.set_title('Anomaly Probability')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes2[0, 0]
    ax.plot(time_index, stl_results['strength_trend'], label='Trend Strength')
    ax.plot(time_index, stl_results['strength_seasonal'], label='Seasonal Strength')
    ax.set_ylabel('Strength')
    ax.set_title('Component Strengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    ax = axes2[0, 1]
    for period in [60, 300]:
        if f'residual_{period}' in stl_results:
            ax.plot(time_index, stl_results[f'residual_{period}'], 
                   alpha=0.7, label=f'{period}s period')
    ax.set_ylabel('Residual')
    ax.set_title('Multi-scale Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    residuals_clean = stl_results['residual'][~np.isnan(stl_results['residual'])]
    ax.hist(residuals_clean, bins=50, density=True, alpha=0.7, color='blue')

    mu, sigma = np.mean(residuals_clean), np.std(residuals_clean)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Normal', 'Anomaly'],
           yticklabels=['Normal', 'Anomaly'],
           xlabel='Predicted',
           ylabel='True')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def predict_with_stl_xgb(new_data_path, stl_config_path, xgb_model_path,
                        scaler_path, config_path, output_path,
                        speed_col='A2:MCPGSpeed', timestamp_col='Time_stamp'):
    """
    Make predictions on new data using STL-XGBoost model.
    """
    print("\n" + "="*60)
    print("PREDICTION WITH STL-XGBOOST MODEL")
    print("="*60)

    stl_config = joblib.load(stl_config_path)
    xgb_model = joblib.load(xgb_model_path)
    scaler = joblib.load(scaler_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    stl_decomposer = CausalSTLDecomposer(
        seasonal_periods=stl_config['seasonal_periods'],
        trend_window=stl_config['trend_window'],
        seasonal_window=stl_config['seasonal_window'],
        robust=stl_config['robust']
    )

    new_df = pd.read_csv(new_data_path, parse_dates=[timestamp_col])
    new_df = new_df.rename(columns={speed_col: 'speed', timestamp_col: 'indo_time'})
    new_df = new_df.sort_values('indo_time').reset_index(drop=True)
    
    print(f"\nLoaded {len(new_df)} samples")
    print(f"Speed range: [{new_df['speed'].min():.2f}, {new_df['speed'].max():.2f}]")

    print("\nApplying STL decomposition...")
    stl_results = stl_decomposer.fit_transform_windowed(
        new_df['speed'].values,
        window_size=7200
    )

    print("Engineering features...")
    X_new = engineer_stl_features(
        stl_results,
        new_df['speed'].values,
        stl_config['window_sizes']
    )

    X_new['hour'] = pd.to_datetime(new_df['indo_time']).dt.hour
    X_new['minute'] = pd.to_datetime(new_df['indo_time']).dt.minute
    X_new['hour_sin'] = np.sin(2 * np.pi * X_new['hour'] / 24)
    X_new['hour_cos'] = np.cos(2 * np.pi * X_new['hour'] / 24)
    X_new['minute_sin'] = np.sin(2 * np.pi * X_new['minute'] / 60)
    X_new['minute_cos'] = np.cos(2 * np.pi * X_new['minute'] / 60)

    min_samples = config['min_samples']
    X_new = X_new.iloc[min_samples:].reset_index(drop=True)
    new_df_trimmed = new_df.iloc[min_samples:].reset_index(drop=True)

    X_new = X_new[config['feature_names']]

    X_new = X_new.replace([np.inf, -np.inf], np.nan)
    train_medians = pd.Series(config['train_medians'])
    X_new = X_new.fillna(train_medians)

    for col in X_new.columns:
        if col in config['train_percentiles']:
            p001 = config['train_percentiles'][col]['p001']
            p999 = config['train_percentiles'][col]['p999']
            X_new[col] = X_new[col].clip(p001, p999)
    
    X_new_scaled = scaler.transform(X_new)

    print("Making predictions...")
    predictions = xgb_model.predict(X_new_scaled)
    probabilities = xgb_model.predict_proba(X_new_scaled)[:, 1]

    results_df = pd.DataFrame({
        'timestamp': new_df_trimmed['indo_time'],
        'speed': new_df_trimmed['speed'],
        'anomaly_probability': probabilities,
        'predicted_label': predictions,
        'anomaly_status': ['Anomaly' if p == 1 else 'Normal' for p in predictions],
        'stl_residual': stl_results['residual'][min_samples:],
        'stl_trend': stl_results['trend'][min_samples:],
        'residual_zscore': X_new['residual_zscore_mad_60'].values
    })

    anomaly_rate = predictions.mean()
    print(f"\nPrediction Summary:")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Detected anomalies: {predictions.sum()} ({anomaly_rate:.2%})")
    print(f"  Expected rate: {config['anomaly_ratio_train']:.2%}")

    anomaly_residuals = results_df.loc[predictions == 1, 'stl_residual'].abs()
    if len(anomaly_residuals) > 0:
        print(f"\nAnomaly Residual Statistics:")
        print(f"  Mean absolute residual: {anomaly_residuals.mean():.3f}")
        print(f"  Max absolute residual: {anomaly_residuals.max():.3f}")

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results_df


if __name__ == "__main__":
    file_path = 'Data\labelled_1_23.csv'
    data_df = pd.read_csv(file_path, parse_dates=['indo_time'])

    stl_decomposer, xgb_model, scaler, config = create_stl_xgb_detector(
        df=data_df,
        speed_col='speed',
        label_col='pred_label',
        timestamp_col='indo_time',
        random_state=42,
        save_path_prefix='stl_xgb_anomaly'
    )
    
    print("\nSTL-XGBoost model training complete!")

    results = predict_with_stl_xgb(
        new_data_path='Data/input_data.csv',
        stl_config_path='stl_xgb_anomaly_stl_config.pkl',
        xgb_model_path='stl_xgb_anomaly_xgb_model.pkl',
        scaler_path='stl_xgb_anomaly_scaler.pkl',
        config_path='stl_xgb_anomaly_config.json',
        output_path='stl_xgb_predictions.csv'
    )