import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FEATURE ENGINEERING
# ============================================
def create_features(df):
    """Create technical indicators for each stock"""
    result = []
    
    for ticker in df['ticker'].unique():
        df_stock = df[df['ticker'] == ticker].copy()
        df_stock = df_stock.sort_values('date')
        
        # Basic features
        df_stock['returns'] = df_stock['Close'].pct_change()
        df_stock['log_returns'] = np.log(df_stock['Close'] / df_stock['Close'].shift(1))
        
        # Moving averages
        df_stock['MA_5'] = df_stock['Close'].rolling(window=5).mean()
        df_stock['MA_10'] = df_stock['Close'].rolling(window=10).mean()
        df_stock['MA_20'] = df_stock['Close'].rolling(window=20).mean()
        df_stock['MA_50'] = df_stock['Close'].rolling(window=50).mean()
        
        # Volatility
        df_stock['volatility_10'] = df_stock['returns'].rolling(window=10).std()
        df_stock['volatility_20'] = df_stock['returns'].rolling(window=20).std()
        
        # RSI
        delta = df_stock['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_stock['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df_stock['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_stock['Close'].ewm(span=26, adjust=False).mean()
        df_stock['MACD'] = exp1 - exp2
        df_stock['Signal_Line'] = df_stock['MACD'].ewm(span=9, adjust=False).mean()
        
        # Price momentum
        df_stock['momentum_5'] = df_stock['Close'] - df_stock['Close'].shift(5)
        df_stock['momentum_10'] = df_stock['Close'] - df_stock['Close'].shift(10)
        
        # Volume features
        df_stock['volume_change'] = df_stock['Volume'].pct_change()
        df_stock['volume_MA_20'] = df_stock['Volume'].rolling(window=20).mean()
        
        # Target: Next day's return (percentage change is better for multi-stock)
        df_stock['target_return'] = df_stock['Close'].pct_change().shift(-1)
        df_stock['target'] = df_stock['Close'].shift(-1)
        
        result.append(df_stock)
    
    return pd.concat(result, ignore_index=True).dropna()

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build LSTM architecture"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ============================================
# MULTI-STOCK TRAINING
# ============================================
def train_multi_stock_model(df, tickers=None, sequence_length=60):
    """
    Train ONE model on MULTIPLE stocks
    
    Parameters:
    - df: DataFrame with all stock data
    - tickers: List of tickers to include (None = all)
    - sequence_length: LSTM sequence length
    """
    print(f"\n{'='*60}")
    print(f"Training Multi-Stock Hybrid Model")
    print(f"{'='*60}\n")
    
    # Filter tickers if specified
    if tickers:
        df = df[df['ticker'].isin(tickers)].copy()
    
    df['date'] = pd.to_datetime(df['date'])
    
    unique_tickers = df['ticker'].unique()
    print(f"Training on {len(unique_tickers)} stocks")
    print(f"Total data points: {len(df)}")
    print(f"Tickers: {list(unique_tickers[:10])}...")
    
    # Feature engineering (does it per stock)
    print("\nCreating features for all stocks...")
    df_features = create_features(df)
    print(f"After feature engineering: {df_features.shape}")
    
    # ==========================================
    # PART A: LSTM MODEL
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL (ALL STOCKS)")
    print("="*60)
    
    lstm_features = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'MACD', 'volatility_10']
    
    # Normalize per stock to handle different price scales
    print("Normalizing data per stock...")
    scaler_lstm = MinMaxScaler()
    lstm_data_list = []
    y_data_list = []
    
    for ticker in df_features['ticker'].unique():
        df_ticker = df_features[df_features['ticker'] == ticker].copy()
        df_ticker = df_ticker.sort_values('date')
        
        if len(df_ticker) < sequence_length + 10:
            continue  # Skip stocks with insufficient data
        
        ticker_data = df_ticker[lstm_features].values
        ticker_scaled = scaler_lstm.fit_transform(ticker_data)
        
        # Create sequences for this stock
        X_ticker, y_ticker = create_sequences(ticker_scaled, sequence_length)
        lstm_data_list.append(X_ticker)
        y_data_list.append(y_ticker)
    
    # Combine all stocks
    X_lstm = np.concatenate(lstm_data_list, axis=0)
    y_lstm = np.concatenate(y_data_list, axis=0)
    
    print(f"Combined LSTM data shape: {X_lstm.shape}")
    
    # Shuffle data (important for multi-stock training)
    shuffle_idx = np.random.permutation(len(X_lstm))
    X_lstm = X_lstm[shuffle_idx]
    y_lstm = y_lstm[shuffle_idx]
    
    # Split train/test
    split_idx = int(0.8 * len(X_lstm))
    X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_lstm_train, y_lstm_test = y_lstm[:split_idx], y_lstm[split_idx:]
    
    print(f"LSTM Train shape: {X_lstm_train.shape}")
    print(f"LSTM Test shape: {X_lstm_test.shape}")
    
    # Build and train LSTM
    lstm_model = build_lstm_model((X_lstm_train.shape[1], X_lstm_train.shape[2]))
    
    print("\nTraining LSTM on all stocks...")
    history = lstm_model.fit(
        X_lstm_train, y_lstm_train,
        batch_size=64,
        epochs=30,
        validation_split=0.1,
        verbose=1
    )
    
    # LSTM predictions
    lstm_pred_train = lstm_model.predict(X_lstm_train, verbose=0).flatten()
    lstm_pred_test = lstm_model.predict(X_lstm_test, verbose=0).flatten()
    
    lstm_train_rmse = np.sqrt(mean_squared_error(y_lstm_train, lstm_pred_train))
    lstm_test_rmse = np.sqrt(mean_squared_error(y_lstm_test, lstm_pred_test))
    lstm_train_mae = mean_absolute_error(y_lstm_train, lstm_pred_train)
    lstm_test_mae = mean_absolute_error(y_lstm_test, lstm_pred_test)
    
    print(f"\n✓ LSTM Train RMSE: {lstm_train_rmse:.4f}")
    print(f"✓ LSTM Test RMSE: {lstm_test_rmse:.4f}")
    print(f"✓ LSTM Train MAE: {lstm_train_mae:.4f}")
    print(f"✓ LSTM Test MAE: {lstm_test_mae:.4f}")
    
    # ==========================================
    # PART B: XGBOOST MODEL
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL (ALL STOCKS)")
    print("="*60)
    
    xgb_features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA_5', 'MA_10', 'MA_20', 'MA_50',
                    'volatility_10', 'volatility_20', 'RSI', 'MACD',
                    'momentum_5', 'momentum_10', 'volume_change']
    
    # For multi-stock, use percentage returns as target
    df_xgb = df_features[xgb_features + ['target_return']].dropna()
    
    X_xgb = df_xgb[xgb_features].values
    y_xgb = df_xgb['target_return'].values
    
    # Shuffle data
    shuffle_idx = np.random.permutation(len(X_xgb))
    X_xgb = X_xgb[shuffle_idx]
    y_xgb = y_xgb[shuffle_idx]
    
    # Split train/test
    split_idx_xgb = int(0.8 * len(X_xgb))
    X_xgb_train, X_xgb_test = X_xgb[:split_idx_xgb], X_xgb[split_idx_xgb:]
    y_xgb_train, y_xgb_test = y_xgb[:split_idx_xgb], y_xgb[split_idx_xgb:]
    
    print(f"XGBoost Train shape: {X_xgb_train.shape}")
    print(f"XGBoost Test shape: {X_xgb_test.shape}")
    
    # Train XGBoost
    print("\nTraining XGBoost on all stocks...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_xgb_train, y_xgb_train, verbose=False)
    
    # XGBoost predictions
    xgb_pred_train = xgb_model.predict(X_xgb_train)
    xgb_pred_test = xgb_model.predict(X_xgb_test)
    
    xgb_train_rmse = np.sqrt(mean_squared_error(y_xgb_train, xgb_pred_train))
    xgb_test_rmse = np.sqrt(mean_squared_error(y_xgb_test, xgb_pred_test))
    xgb_train_mae = mean_absolute_error(y_xgb_train, xgb_pred_train)
    xgb_test_mae = mean_absolute_error(y_xgb_test, xgb_pred_test)
    
    print(f"\n✓ XGBoost Train RMSE: {xgb_train_rmse:.4f}")
    print(f"✓ XGBoost Test RMSE: {xgb_test_rmse:.4f}")
    print(f"✓ XGBoost Train MAE: {xgb_train_mae:.4f}")
    print(f"✓ XGBoost Test MAE: {xgb_test_mae:.4f}")
    
    # ==========================================
    # ENSEMBLE
    # ==========================================
    print("\n" + "="*60)
    print("CREATING ENSEMBLE MODEL")
    print("="*60)
    
    # Simple weighted average for ensemble
    lstm_weight = 0.6
    xgb_weight = 0.4
    
    print(f"\nEnsemble Weights: LSTM={lstm_weight}, XGBoost={xgb_weight}")
    
    # ==========================================
    # RESULTS SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    results = pd.DataFrame({
        'Model': ['LSTM', 'XGBoost'],
        'Train RMSE': [lstm_train_rmse, xgb_train_rmse],
        'Test RMSE': [lstm_test_rmse, xgb_test_rmse],
        'Train MAE': [lstm_train_mae, xgb_train_mae],
        'Test MAE': [lstm_test_mae, xgb_test_mae]
    })
    
    print("\n", results.to_string(index=False))
    
    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'scaler_lstm': scaler_lstm,
        'results': results,
        'lstm_weight': lstm_weight,
        'xgb_weight': xgb_weight,
        'sequence_length': sequence_length,
        'lstm_features': lstm_features,
        'xgb_features': xgb_features,
        'training_type': 'multi_stock',
        'tickers': list(unique_tickers)
    }

# ============================================
# SINGLE-STOCK TRAINING (Original)
# ============================================
def train_single_stock_model(df, ticker='AAPL', sequence_length=60):
    """
    Train model on SINGLE stock (original approach)
    """
    print(f"\n{'='*60}")
    print(f"Training Single-Stock Model for {ticker}")
    print(f"{'='*60}\n")
    
    df_stock = df[df['ticker'] == ticker].copy()
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock = df_stock.sort_values('date').reset_index(drop=True)
    
    print(f"Data shape: {df_stock.shape}")
    print(f"Date range: {df_stock['date'].min()} to {df_stock['date'].max()}\n")
    
    # Feature engineering
    df_features = create_features(df_stock)
    
    # ... (rest of single-stock training code - same as before)
    # I'll abbreviate this since it's the same as the previous version
    
    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'scaler_lstm': scaler_lstm,
        'results': results,
        'training_type': 'single_stock',
        'ticker': ticker
    }

# ============================================
# MODEL SAVING/LOADING
# ============================================
def save_models(models, model_name='multi_stock', save_dir='saved_models'):
    """Save trained models"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SAVING MODELS")
    print(f"{'='*60}")
    
    # Save LSTM
    lstm_path = os.path.join(model_path, 'lstm_model.h5')
    models['lstm_model'].save(lstm_path)
    print(f"✓ Saved LSTM: {lstm_path}")
    
    # Save XGBoost
    xgb_path = os.path.join(model_path, 'xgb_model.json')
    models['xgb_model'].save_model(xgb_path)
    print(f"✓ Saved XGBoost: {xgb_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_path, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(models['scaler_lstm'], f)
    print(f"✓ Saved scaler: {scaler_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'training_type': models.get('training_type', 'unknown'),
        'lstm_weight': models['lstm_weight'],
        'xgb_weight': models['xgb_weight'],
        'sequence_length': models['sequence_length'],
        'lstm_features': models['lstm_features'],
        'xgb_features': models['xgb_features']
    }
    
    if models.get('training_type') == 'single_stock':
        metadata['ticker'] = models['ticker']
    else:
        metadata['tickers'] = models.get('tickers', [])
    
    metadata_path = os.path.join(model_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Saved metadata: {metadata_path}")
    
    print(f"\n✅ All models saved to: {model_path}")
    return model_path

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('stock_data_5years.csv')
    
    print(f"\nTotal data shape: {df.shape}")
    print(f"Available tickers: {df['ticker'].nunique()}")
    print(f"Sample tickers: {list(df['ticker'].unique()[:20])}")
    
    print("\n" + "="*60)
    print("CHOOSE TRAINING MODE")
    print("="*60)
    print("1. Single Stock (one model per stock)")
    print("2. Multi Stock (one model for all stocks)")
    
    mode = input("\nEnter 1 or 2: ").strip()
    
    if mode == "1":
        # SINGLE STOCK MODE
        ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
        
        if ticker not in df['ticker'].values:
            print(f"❌ Ticker {ticker} not found in data!")
            exit()
        
        models = train_single_stock_model(df, ticker=ticker)
        model_path = save_models(models, model_name=ticker)
        
    elif mode == "2":
        # MULTI STOCK MODE
        use_all = input("Train on all stocks? (y/n): ").strip().lower()
        
        if use_all == 'y':
            models = train_multi_stock_model(df)
            model_path = save_models(models, model_name='all_stocks')
        else:
            # Train on subset
            n_stocks = int(input("How many stocks? "))
            tickers = df['ticker'].unique()[:n_stocks]
            print(f"Training on: {list(tickers)}")
            
            models = train_multi_stock_model(df, tickers=tickers)
            model_path = save_models(models, model_name=f'{n_stocks}_stocks')
    
    else:
        print("❌ Invalid choice!")
        exit()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)