import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import itertools
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# ENHANCED FEATURE ENGINEERING
# ============================================
# ============================================
# ENHANCED FEATURE ENGINEERING (FIXED)
# ============================================
def create_enhanced_features(df):
    """Create advanced technical indicators"""
    df = df.copy()
    df = df.sort_values('date')
    
    # Basic features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Multiple timeframe moving averages
    for window in [3, 5, 10, 20, 50, 100, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
    
    # Exponential moving averages
    for span in [12, 26, 50]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    # Volatility measures - CREATE ALL FIRST
    for window in [5, 10, 20, 30]:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
    
    # Now create volatility ratios AFTER all volatilities exist
    for window in [5, 20, 30]:
        if window != 10:  # Skip volatility_10/volatility_10
            df[f'volatility_{window}_ratio'] = df['volatility_10'] / df[f'volatility_{window}']
    
    # RSI (multiple timeframes)
    for window in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    for window in [20, 50]:
        df[f'BB_middle_{window}'] = df['Close'].rolling(window=window).mean()
        bb_std = df['Close'].rolling(window=window).std()
        df[f'BB_upper_{window}'] = df[f'BB_middle_{window}'] + 2*bb_std
        df[f'BB_lower_{window}'] = df[f'BB_middle_{window}'] - 2*bb_std
        df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / df[f'BB_middle_{window}']
        df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}'])
    
    # Momentum indicators
    for window in [5, 10, 20]:
        df[f'momentum_{window}'] = df['Close'] - df['Close'].shift(window)
        df[f'momentum_{window}_pct'] = df['Close'].pct_change(window)
    
    # Rate of Change
    for window in [5, 10, 20]:
        df[f'ROC_{window}'] = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    
    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    for window in [5, 10, 20]:
        df[f'volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_MA_{window}']
    
    # Price channels
    for window in [20, 50]:
        df[f'high_{window}'] = df['High'].rolling(window=window).max()
        df[f'low_{window}'] = df['Low'].rolling(window=window).min()
        df[f'channel_position_{window}'] = (df['Close'] - df[f'low_{window}']) / (df[f'high_{window}'] - df[f'low_{window}'])
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    
    # Stochastic Oscillator
    for window in [14, 21]:
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        df[f'Stoch_{window}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    # Target
    df['target'] = df['Close'].shift(-1)
    df['target_return'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna()
# ============================================
# FEATURE SELECTION
# ============================================
def select_best_features(df, target_col='target_return', top_n=30):
    """Select most important features using correlation and XGBoost"""
    feature_cols = [col for col in df.columns if col not in ['date', 'ticker', 'target', 'target_return']]
    
    # Correlation-based selection
    correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    top_corr_features = correlations.head(top_n).index.tolist()
    
    print(f"\nTop {top_n} features by correlation:")
    for i, (feat, corr) in enumerate(correlations.head(top_n).items(), 1):
        print(f"  {i}. {feat}: {corr:.4f}")
    
    return top_corr_features

# ============================================
# CREATE SEQUENCES
# ============================================
def create_sequences(data, seq_length=60):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ============================================
# BUILD OPTIMIZED LSTM MODEL
# ============================================
def build_lstm_model(input_shape, config):
    """Build LSTM with configurable architecture"""
    model = Sequential()
    
    # First layer
    if config['use_bidirectional']:
        model.add(Bidirectional(LSTM(config['lstm_units'][0], return_sequences=True), 
                               input_shape=input_shape))
    else:
        model.add(LSTM(config['lstm_units'][0], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(config['dropout']))
    
    # Hidden layers
    for units in config['lstm_units'][1:-1]:
        if config['use_bidirectional']:
            model.add(Bidirectional(LSTM(units, return_sequences=True)))
        else:
            model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(config['dropout']))
    
    # Last LSTM layer
    if config['use_bidirectional']:
        model.add(Bidirectional(LSTM(config['lstm_units'][-1], return_sequences=False)))
    else:
        model.add(LSTM(config['lstm_units'][-1], return_sequences=False))
    model.add(Dropout(config['dropout']))
    
    # Dense layers
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dropout'] * 0.5))
    
    # Output
    model.add(Dense(1))
    
    # Compile
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# ============================================
# HYPERPARAMETER OPTIMIZATION
# ============================================
def optimize_hyperparameters(df, ticker, feature_set='basic'):
    """Find best hyperparameters through grid search"""
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER OPTIMIZATION FOR {ticker}")
    print(f"{'='*60}\n")
    
    # Prepare data
    df_stock = df[df['ticker'] == ticker].copy()
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock = df_stock.sort_values('date').reset_index(drop=True)
    
    df_features = create_enhanced_features(df_stock)
    
    # Select features
    if feature_set == 'basic':
        lstm_features = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI_14', 'MACD', 'volatility_10']
    elif feature_set == 'medium':
        lstm_features = ['Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 'RSI_14', 'RSI_7', 
                        'MACD', 'volatility_10', 'volatility_20', 'momentum_10', 'ROC_10']
    else:  # advanced
        lstm_features = select_best_features(df_features, top_n=20)
    
    print(f"Using {len(lstm_features)} features: {lstm_features[:5]}...")
    
    # Hyperparameter search space
    configs = [
        # Configuration 1: Small & Deep
        {
            'name': 'Small_Deep',
            'lstm_units': [64, 64, 32],
            'dense_units': [32],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'sequence_length': 60,
            'use_bidirectional': False
        },
        # Configuration 2: Large & Shallow
        {
            'name': 'Large_Shallow',
            'lstm_units': [128, 64],
            'dense_units': [32],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50,
            'sequence_length': 60,
            'use_bidirectional': False
        },
        # Configuration 3: Bidirectional
        {
            'name': 'Bidirectional',
            'lstm_units': [100, 50],
            'dense_units': [25],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'sequence_length': 60,
            'use_bidirectional': True
        },
        # Configuration 4: High Dropout (for overfitting)
        {
            'name': 'High_Dropout',
            'lstm_units': [128, 64, 32],
            'dense_units': [32, 16],
            'dropout': 0.5,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 50,
            'sequence_length': 60,
            'use_bidirectional': False
        },
        # Configuration 5: Long Sequence
        {
            'name': 'Long_Sequence',
            'lstm_units': [100, 50],
            'dense_units': [25],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'sequence_length': 90,
            'use_bidirectional': False
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing Configuration: {config['name']}")
        print(f"{'='*60}")
        print(f"LSTM Units: {config['lstm_units']}")
        print(f"Dropout: {config['dropout']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Sequence Length: {config['sequence_length']}")
        print(f"Bidirectional: {config['use_bidirectional']}")
        
        try:
            # Prepare data
            lstm_data = df_features[lstm_features].values
            scaler = MinMaxScaler()
            lstm_scaled = scaler.fit_transform(lstm_data)
            
            X, y = create_sequences(lstm_scaled, config['sequence_length'])
            
            # Split
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]), config)
            
            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            
            # Train
            print("\nTraining...")
            history = model.fit(
                X_train, y_train,
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                validation_split=0.1,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluate
            train_pred = model.predict(X_train, verbose=0).flatten()
            test_pred = model.predict(X_test, verbose=0).flatten()
            
            # Inverse transform
            def inverse_transform(predictions, scaler):
                dummy = np.zeros((len(predictions), scaler.n_features_in_))
                dummy[:, 0] = predictions
                return scaler.inverse_transform(dummy)[:, 0]
            
            train_pred_real = inverse_transform(train_pred, scaler)
            test_pred_real = inverse_transform(test_pred, scaler)
            y_train_real = inverse_transform(y_train, scaler)
            y_test_real = inverse_transform(y_test, scaler)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_real, train_pred_real))
            test_rmse = np.sqrt(mean_squared_error(y_test_real, test_pred_real))
            
            # Directional accuracy
            train_dir_acc = np.mean((np.diff(y_train_real) > 0) == (np.diff(train_pred_real) > 0)) * 100
            test_dir_acc = np.mean((np.diff(y_test_real) > 0) == (np.diff(test_pred_real) > 0)) * 100
            
            # Overfitting check
            overfit_score = train_rmse / test_rmse  # Should be close to 1
            
            print(f"\n✓ Train RMSE: ${train_rmse:.2f}")
            print(f"✓ Test RMSE: ${test_rmse:.2f}")
            print(f"✓ Train Dir Acc: {train_dir_acc:.2f}%")
            print(f"✓ Test Dir Acc: {test_dir_acc:.2f}%")
            print(f"✓ Overfit Score: {overfit_score:.2f} (1.0 = perfect)")
            
            results.append({
                'config': config['name'],
                'test_rmse': test_rmse,
                'test_dir_acc': test_dir_acc,
                'train_rmse': train_rmse,
                'overfit_score': overfit_score,
                'config_details': config
            })
            
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
    
    # Find best configuration
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_dir_acc', ascending=False)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}\n")
    print(results_df[['config', 'test_dir_acc', 'test_rmse', 'overfit_score']].to_string(index=False))
    
    best_config = results_df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION: {best_config['config']}")
    print(f"   Directional Accuracy: {best_config['test_dir_acc']:.2f}%")
    print(f"   Test RMSE: ${best_config['test_rmse']:.2f}")
    
    return best_config['config_details'], results_df

# ============================================
# OPTIMIZE XGBOOST
# ============================================
def optimize_xgboost(df, ticker, feature_set='basic'):
    """Optimize XGBoost hyperparameters"""
    print(f"\n{'='*60}")
    print(f"XGBOOST OPTIMIZATION FOR {ticker}")
    print(f"{'='*60}\n")
    
    df_stock = df[df['ticker'] == ticker].copy()
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_features = create_enhanced_features(df_stock)
    
    # Select features
    if feature_set == 'basic':
        xgb_features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA_5', 'MA_20', 'RSI_14', 'MACD', 'volatility_10']
    else:
        xgb_features = select_best_features(df_features, top_n=30)
    
    df_xgb = df_features[xgb_features + ['target_return']].dropna()
    X = df_xgb[xgb_features].values
    y = df_xgb['target_return'].values
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test configurations
    configs = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.01},
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.01},
        {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.03},
    ]
    
    results = []
    
    for config in configs:
        model = xgb.XGBRegressor(**config, random_state=42)
        model.fit(X_train, y_train, verbose=False)
        
        test_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Convert returns to prices for directional accuracy
        current_prices = df_features['Close'].values[split_idx:-1]
        actual_prices = current_prices * (1 + y_test[:-1])
        pred_prices = current_prices * (1 + test_pred[:-1])
        
        dir_acc = np.mean((np.diff(actual_prices) > 0) == (np.diff(pred_prices) > 0)) * 100
        
        results.append({
            'config': str(config),
            'test_rmse': test_rmse,
            'dir_acc': dir_acc,
            'params': config
        })
        
        print(f"Config: {config}")
        print(f"  RMSE: {test_rmse:.4f}, Dir Acc: {dir_acc:.2f}%\n")
    
    best = max(results, key=lambda x: x['dir_acc'])
    print(f"🏆 Best XGBoost Config: {best['params']}")
    print(f"   Directional Accuracy: {best['dir_acc']:.2f}%")
    
    return best['params']

# ============================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ============================================
def optimize_ensemble_weights(lstm_predictions, xgb_predictions, actual_values):
    """Find optimal ensemble weights"""
    print(f"\n{'='*60}")
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print(f"{'='*60}\n")
    
    best_accuracy = 0
    best_weights = (0.5, 0.5)
    
    # Try different weight combinations
    for lstm_weight in np.arange(0, 1.1, 0.1):
        xgb_weight = 1 - lstm_weight
        
        ensemble_pred = lstm_weight * lstm_predictions + xgb_weight * xgb_predictions
        
        # Directional accuracy
        dir_acc = np.mean((np.diff(actual_values) > 0) == (np.diff(ensemble_pred) > 0)) * 100
        
        if dir_acc > best_accuracy:
            best_accuracy = dir_acc
            best_weights = (lstm_weight, xgb_weight)
        
        print(f"LSTM: {lstm_weight:.1f}, XGBoost: {xgb_weight:.1f} → Accuracy: {dir_acc:.2f}%")
    
    print(f"\n🏆 Best Weights: LSTM={best_weights[0]:.1f}, XGBoost={best_weights[1]:.1f}")
    print(f"   Directional Accuracy: {best_accuracy:.2f}%")
    
    return best_weights

# ============================================
# MAIN FINE-TUNING PIPELINE
# ============================================
def main():
    print("="*60)
    print("MODEL FINE-TUNING")
    print("="*60)
    
    # Load data
    df = pd.read_csv('stock_data_5years.csv')
    
    # Select ticker
    print("\nAvailable tickers:")
    tickers = sorted(df['ticker'].unique())
    for i in range(0, len(tickers), 5):
        print("  " + "  ".join(f"{t:<10}" for t in tickers[i:i+5]))
    
    ticker = input("\nEnter ticker to fine-tune: ").strip().upper()
    
    if ticker not in tickers:
        print("❌ Invalid ticker!")
        return
    
    # Feature set selection
    print("\nFeature set:")
    print("1. Basic (7 features) - Fast")
    print("2. Medium (12 features) - Balanced")
    print("3. Advanced (20+ features) - Slow but thorough")
    
    feature_choice = input("Select (1-3): ").strip()
    feature_set = ['basic', 'medium', 'advanced'][int(feature_choice)-1]
    
    # Run optimization
    print(f"\n🚀 Starting fine-tuning for {ticker}...")
    
    # 1. Optimize LSTM
    best_lstm_config, lstm_results = optimize_hyperparameters(df, ticker, feature_set)
    
    # 2. Optimize XGBoost
    best_xgb_config = optimize_xgboost(df, ticker, feature_set)
    
    # 3. Save results
    results = {
        'ticker': ticker,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'best_lstm_config': best_lstm_config,
        'best_xgb_config': best_xgb_config,
        'feature_set': feature_set
    }
    
    results_file = f'finetuning_results_{ticker}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Saved fine-tuning results to: {results_file}")
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use the best configuration to retrain your model")
    print("2. The optimal hyperparameters are saved in the JSON file")
    print("3. Update your training script with these parameters")

if __name__ == "__main__":
    main()