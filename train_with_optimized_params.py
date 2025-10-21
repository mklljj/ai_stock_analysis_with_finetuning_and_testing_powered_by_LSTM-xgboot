import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the feature engineering from finetune_model.py
def create_enhanced_features(df):
    """Create advanced technical indicators"""
    df = df.copy()
    df = df.sort_values('date')
    
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    for window in [3, 5, 10, 20, 50, 100, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
    
    for span in [12, 26, 50]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    for window in [5, 10, 20, 30]:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
    
    for window in [5, 20, 30]:
        if window != 10:
            df[f'volatility_{window}_ratio'] = df['volatility_10'] / df[f'volatility_{window}']
    
    for window in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    for window in [20, 50]:
        df[f'BB_middle_{window}'] = df['Close'].rolling(window=window).mean()
        bb_std = df['Close'].rolling(window=window).std()
        df[f'BB_upper_{window}'] = df[f'BB_middle_{window}'] + 2*bb_std
        df[f'BB_lower_{window}'] = df[f'BB_middle_{window}'] - 2*bb_std
        df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / df[f'BB_middle_{window}']
        df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}'])
    
    for window in [5, 10, 20]:
        df[f'momentum_{window}'] = df['Close'] - df['Close'].shift(window)
        df[f'momentum_{window}_pct'] = df['Close'].pct_change(window)
        df[f'ROC_{window}'] = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    
    df['volume_change'] = df['Volume'].pct_change()
    for window in [5, 10, 20]:
        df[f'volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_MA_{window}']
    
    for window in [20, 50]:
        df[f'high_{window}'] = df['High'].rolling(window=window).max()
        df[f'low_{window}'] = df['Low'].rolling(window=window).min()
        df[f'channel_position_{window}'] = (df['Close'] - df[f'low_{window}']) / (df[f'high_{window}'] - df[f'low_{window}'])
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    
    for window in [14, 21]:
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        df[f'Stoch_{window}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    df['target'] = df['Close'].shift(-1)
    df['target_return'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna()

def select_best_features(df, target_col='target_return', top_n=20):
    """Select most important features"""
    feature_cols = [col for col in df.columns if col not in ['date', 'ticker', 'target', 'target_return']]
    correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    return correlations.head(top_n).index.tolist()

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_optimized_model(ticker, config_file='finetuning_results_PFE.json'):
    """Train model with optimized parameters"""
    
    print(f"{'='*60}")
    print(f"TRAINING OPTIMIZED MODEL FOR {ticker}")
    print(f"{'='*60}\n")
    
    # Load optimized configuration
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        print("   Run finetune_model.py first!")
        return
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    lstm_config = config_data['best_lstm_config']
    xgb_config = config_data['best_xgb_config']
    
    print("Using optimized configuration:")
    print(f"  LSTM: {lstm_config['name']}")
    print(f"  Units: {lstm_config['lstm_units']}")
    print(f"  Dropout: {lstm_config['dropout']}")
    print(f"  XGBoost: {xgb_config}")
    
    # Load data
    df = pd.read_csv('stock_data_5years.csv')
    df_stock = df[df['ticker'] == ticker].copy()
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock = df_stock.sort_values('date').reset_index(drop=True)
    
    print(f"\nData: {len(df_stock)} rows")
    
    # Feature engineering
    print("Creating enhanced features...")
    df_features = create_enhanced_features(df_stock)
    
    # Select best features
    lstm_features = select_best_features(df_features, top_n=20)
    print(f"Selected {len(lstm_features)} LSTM features")
    
    # Prepare LSTM data
    lstm_data = df_features[lstm_features].values
    scaler = MinMaxScaler()
    lstm_scaled = scaler.fit_transform(lstm_data)
    
    X, y = create_sequences(lstm_scaled, lstm_config['sequence_length'])
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Build LSTM with optimized config
    print("\nBuilding LSTM model...")
    model = Sequential()
    
    units = lstm_config['lstm_units']
    dropout = lstm_config['dropout']
    use_bi = lstm_config.get('use_bidirectional', False)
    
    if use_bi:
        model.add(Bidirectional(LSTM(units[0], return_sequences=len(units) > 1), 
                               input_shape=(X_train.shape[1], X_train.shape[2])))
    else:
        model.add(LSTM(units[0], return_sequences=len(units) > 1, 
                      input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    
    for i, u in enumerate(units[1:]):
        return_seq = i < len(units) - 2
        if use_bi:
            model.add(Bidirectional(LSTM(u, return_sequences=return_seq)))
        else:
            model.add(LSTM(u, return_sequences=return_seq))
        model.add(Dropout(dropout))
    
    for du in lstm_config.get('dense_units', [25]):
        model.add(Dense(du, activation='relu'))
        model.add(Dropout(dropout * 0.5))
    
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=lstm_config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Train LSTM
    print("Training LSTM...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    model.fit(
        X_train, y_train,
        batch_size=lstm_config['batch_size'],
        epochs=lstm_config['epochs'],
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate LSTM
    test_pred = model.predict(X_test, verbose=0).flatten()
    
    def inverse_transform(predictions, scaler):
        dummy = np.zeros((len(predictions), scaler.n_features_in_))
        dummy[:, 0] = predictions
        return scaler.inverse_transform(dummy)[:, 0]
    
    test_pred_real = inverse_transform(test_pred, scaler)
    y_test_real = inverse_transform(y_test, scaler)
    
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, test_pred_real))
    lstm_dir_acc = np.mean((np.diff(y_test_real) > 0) == (np.diff(test_pred_real) > 0)) * 100
    
    print(f"\n✓ LSTM Test RMSE: ${lstm_rmse:.2f}")
    print(f"✓ LSTM Directional Accuracy: {lstm_dir_acc:.2f}%")
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_features = select_best_features(df_features, top_n=30)
    df_xgb = df_features[xgb_features + ['target']].dropna()
    
    X_xgb = df_xgb[xgb_features].values
    y_xgb = df_xgb['target'].values
    
    split_idx_xgb = int(0.8 * len(X_xgb))
    X_xgb_train, X_xgb_test = X_xgb[:split_idx_xgb], X_xgb[split_idx_xgb:]
    y_xgb_train, y_xgb_test = y_xgb[:split_idx_xgb], y_xgb[split_idx_xgb:]
    
    xgb_model = xgb.XGBRegressor(**xgb_config, random_state=42)
    xgb_model.fit(X_xgb_train, y_xgb_train, verbose=False)
    
    xgb_pred_test = xgb_model.predict(X_xgb_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_xgb_test, xgb_pred_test))
    xgb_dir_acc = np.mean((np.diff(y_xgb_test) > 0) == (np.diff(xgb_pred_test) > 0)) * 100
    
    print(f"✓ XGBoost Test RMSE: ${xgb_rmse:.2f}")
    print(f"✓ XGBoost Directional Accuracy: {xgb_dir_acc:.2f}%")
    
    # Save models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'saved_models/{ticker}_optimized_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(os.path.join(save_dir, 'lstm_model.h5'))
    xgb_model.save_model(os.path.join(save_dir, 'xgb_model.json'))
    
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'ticker': ticker,
        'timestamp': timestamp,
        'lstm_weight': 0.6,
        'xgb_weight': 0.4,
        'sequence_length': lstm_config['sequence_length'],
        'lstm_features': lstm_features,
        'xgb_features': xgb_features,
        'training_type': 'single_stock_optimized',
        'lstm_config': lstm_config,
        'xgb_config': xgb_config
    }
    
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Saved optimized model to: {save_dir}")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nNow test with: python test_accuracy.py")
    print(f"Select model: {ticker}_optimized_{timestamp}")

if __name__ == "__main__":
    print("Train model with optimized parameters")
    print("="*60)
    
    ticker = input("Enter ticker (must match fine-tuning file): ").strip().upper()
    config_file = f'finetuning_results_{ticker}.json'
    
    train_optimized_model(ticker, config_file)