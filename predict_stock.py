import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import xgboost as xgb
import pickle
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FEATURE ENGINEERING (same as training)
# ============================================
def create_features(df):
    """Create technical indicators"""
    df = df.copy()
    df = df.sort_values('date')
    
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    
    df['target'] = df['Close'].shift(-1)
    df['target_return'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna()

# ============================================
# LOAD SAVED MODEL (FIXED)
# ============================================
def load_saved_model(model_path):
    """Load trained model from disk"""
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    
    # Load LSTM with compile=False to avoid loss function issues
    lstm_path = os.path.join(model_path, 'lstm_model.h5')
    try:
        lstm_model = load_model(lstm_path, compile=False)
        print(f"✓ Loaded LSTM model (without compilation)")
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(f"✓ Recompiled LSTM model")
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")
        raise
    
    # Load XGBoost
    xgb_path = os.path.join(model_path, 'xgb_model.json')
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    print(f"✓ Loaded XGBoost model")
    
    # Load scaler
    scaler_path = os.path.join(model_path, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Loaded scaler")
    
    # Load metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata")
    
    training_type = metadata.get('training_type', 'unknown')
    print(f"\nModel type: {training_type}")
    
    if training_type == 'single_stock':
        print(f"Trained on: {metadata.get('ticker', 'unknown')}")
    elif training_type == 'multi_stock':
        num_tickers = len(metadata.get('tickers', []))
        print(f"Trained on {num_tickers} stocks")
    
    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'metadata': metadata
    }

# ============================================
# MAKE PREDICTION (FIXED FOR MULTI-STOCK)
# ============================================
def predict_stock_price(model_dict, stock_data, ticker):
    """
    Predict next day's stock price
    
    Parameters:
    - model_dict: Dictionary from load_saved_model()
    - stock_data: DataFrame with historical data for the stock
    - ticker: Stock ticker symbol
    
    Returns:
    - Dictionary with predictions
    """
    print(f"\n{'='*60}")
    print(f"PREDICTING: {ticker}")
    print(f"{'='*60}")
    
    metadata = model_dict['metadata']
    sequence_length = metadata['sequence_length']
    lstm_features = metadata['lstm_features']
    xgb_features = metadata['xgb_features']
    training_type = metadata.get('training_type', 'unknown')
    
    # Prepare data
    stock_data = stock_data.copy()
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date')
    
    print(f"Data range: {stock_data['date'].min()} to {stock_data['date'].max()}")
    print(f"Total days: {len(stock_data)}")
    
    # Feature engineering
    df_features = create_features(stock_data)
    
    if len(df_features) < sequence_length:
        print(f"❌ Error: Need at least {sequence_length} days of data!")
        print(f"   Currently have: {len(df_features)} days")
        return None
    
    current_price = df_features['Close'].iloc[-1]
    current_date = df_features['date'].iloc[-1]
    
    print(f"\nCurrent price: ${current_price:.2f}")
    print(f"Current date: {current_date.date()}")
    print(f"Model type: {training_type}")
    
    # ==========================================
    # LSTM PREDICTION
    # ==========================================
    lstm_pred = None
    try:
        lstm_data = df_features[lstm_features].values
        scaler = model_dict['scaler']
        
        # For multi-stock, we need to fit the scaler on THIS stock's data
        if training_type == 'multi_stock':
            # Create a new scaler and fit on current stock
            from sklearn.preprocessing import MinMaxScaler
            stock_scaler = MinMaxScaler()
            lstm_scaled = stock_scaler.fit_transform(lstm_data)
        else:
            lstm_scaled = scaler.transform(lstm_data)
            stock_scaler = scaler
        
        # Get last sequence
        last_sequence = lstm_scaled[-sequence_length:]
        last_sequence = last_sequence.reshape(1, sequence_length, len(lstm_features))
        
        # Predict
        lstm_pred_scaled = model_dict['lstm_model'].predict(last_sequence, verbose=0)[0][0]
        
        # Inverse transform
        dummy = np.zeros((1, stock_scaler.n_features_in_))
        dummy[0, 0] = lstm_pred_scaled
        lstm_pred = stock_scaler.inverse_transform(dummy)[0, 0]
        
        print(f"\n✓ LSTM prediction: ${lstm_pred:.2f}")
    except Exception as e:
        print(f"\n❌ LSTM prediction failed: {e}")
        lstm_pred = None
    
    # ==========================================
    # XGBOOST PREDICTION
    # ==========================================
    xgb_pred = None
    try:
        xgb_data = df_features[xgb_features].iloc[-1:].values
        xgb_pred_raw = model_dict['xgb_model'].predict(xgb_data)[0]
        
        # For multi-stock models, XGBoost predicts percentage return
        if training_type == 'multi_stock':
            # Convert percentage return to price
            xgb_pred = current_price * (1 + xgb_pred_raw)
            print(f"✓ XGBoost predicted return: {xgb_pred_raw*100:.2f}%")
            print(f"✓ XGBoost predicted price: ${xgb_pred:.2f}")
        else:
            # For single-stock, it predicts absolute price
            xgb_pred = xgb_pred_raw
            print(f"✓ XGBoost prediction: ${xgb_pred:.2f}")
            
    except Exception as e:
        print(f"\n❌ XGBoost prediction failed: {e}")
        xgb_pred = None
    
    # ==========================================
    # ENSEMBLE PREDICTION
    # ==========================================
    lstm_weight = metadata['lstm_weight']
    xgb_weight = metadata['xgb_weight']
    
    if lstm_pred is not None and xgb_pred is not None:
        ensemble_pred = lstm_weight * lstm_pred + xgb_weight * xgb_pred
        print(f"✓ Ensemble prediction (weights: LSTM={lstm_weight}, XGB={xgb_weight}): ${ensemble_pred:.2f}")
    elif lstm_pred is not None:
        ensemble_pred = lstm_pred
        print(f"⚠️  Using LSTM only: ${ensemble_pred:.2f}")
    elif xgb_pred is not None:
        ensemble_pred = xgb_pred
        print(f"⚠️  Using XGBoost only: ${ensemble_pred:.2f}")
    else:
        print("❌ No predictions available!")
        return None
    
    # Calculate changes
    change = ensemble_pred - current_price
    change_pct = (change / current_price) * 100
    direction = "📈 UP" if change > 0 else "📉 DOWN"
    
    # Simple moving average for context
    ma_5 = df_features['MA_5'].iloc[-1]
    ma_20 = df_features['MA_20'].iloc[-1]
    rsi = df_features['RSI'].iloc[-1]
    
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Stock: {ticker}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price: ${ensemble_pred:.2f}")
    print(f"Expected Change: {direction} ${abs(change):.2f} ({abs(change_pct):.2f}%)")
    print(f"\nTechnical Indicators:")
    print(f"  5-day MA: ${ma_5:.2f}")
    print(f"  20-day MA: ${ma_20:.2f}")
    print(f"  RSI: {rsi:.2f}")
    print(f"{'='*60}")
    
    return {
        'ticker': ticker,
        'current_date': current_date,
        'current_price': current_price,
        'lstm_prediction': lstm_pred,
        'xgboost_prediction': xgb_pred,
        'ensemble_prediction': ensemble_pred,
        'predicted_change': change,
        'predicted_change_pct': change_pct,
        'direction': 'UP' if change > 0 else 'DOWN',
        'ma_5': ma_5,
        'ma_20': ma_20,
        'rsi': rsi
    }

# ============================================
# MAIN INTERACTIVE PREDICTOR
# ============================================
def main():
    print("="*60)
    print("STOCK PRICE PREDICTOR")
    print("="*60)
    
    # Load your stock data
    print("\nLoading stock data...")
    df = pd.read_csv('stock_data_5years.csv')
    available_tickers = sorted(df['ticker'].unique())
    
    print(f"✓ Loaded data for {len(available_tickers)} stocks")
    print(f"\nAvailable tickers: {', '.join(available_tickers[:20])}...")
    
    # Load trained model
    print("\nAvailable models:")
    if os.path.exists('saved_models'):
        models = [d for d in os.listdir('saved_models') if os.path.isdir(os.path.join('saved_models', d))]
        models = sorted(models, reverse=True)  # Most recent first
        
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        if not models:
            print("❌ No saved models found!")
            print("   Please train a model first using the training script.")
            return
        
        model_choice = int(input("\nSelect model number: ")) - 1
        model_path = os.path.join('saved_models', models[model_choice])
    else:
        print("❌ 'saved_models' directory not found!")
        print("   Please train a model first.")
        return
    
    # Load model
    model_dict = load_saved_model(model_path)
    
    # Interactive prediction loop
    while True:
        print("\n" + "="*60)
        ticker = input("\nEnter stock ticker (or 'quit' to exit): ").strip().upper()
        
        if ticker.lower() == 'quit':
            print("\nThank you for using Stock Price Predictor!")
            break
        
        if ticker not in available_tickers:
            print(f"❌ Ticker '{ticker}' not found in dataset!")
            print(f"   Available tickers: {', '.join(available_tickers[:10])}...")
            continue
        
        # Get stock data
        stock_data = df[df['ticker'] == ticker].copy()
        
        # Make prediction
        prediction = predict_stock_price(model_dict, stock_data, ticker)
        
        if prediction is None:
            print("⚠️  Prediction failed. Try another stock.")
            continue
        
        # Ask if user wants to predict another
        another = input("\nPredict another stock? (y/n): ").strip().lower()
        if another != 'y':
            print("\nThank you for using Stock Price Predictor!")
            break

if __name__ == "__main__":
    main()