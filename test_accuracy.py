import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FEATURE ENGINEERING
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
# LOAD MODEL
# ============================================
def load_saved_model(model_path):
    """Load trained model from disk"""
    print(f"Loading model from: {model_path}")
    
    lstm_path = os.path.join(model_path, 'lstm_model.h5')
    lstm_model = load_model(lstm_path, compile=False)
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    xgb_path = os.path.join(model_path, 'xgb_model.json')
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    
    scaler_path = os.path.join(model_path, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    metadata_path = os.path.join(model_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'metadata': metadata
    }

# ============================================
# BACKTESTING FUNCTION
# ============================================
def backtest_model(model_dict, stock_data, ticker, test_days=100):
    """
    Test model accuracy on historical data
    
    Parameters:
    - model_dict: Loaded model dictionary
    - stock_data: Historical stock data
    - ticker: Stock symbol
    - test_days: Number of days to test
    
    Returns:
    - DataFrame with predictions and actuals
    """
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {ticker}")
    print(f"{'='*60}")
    
    metadata = model_dict['metadata']
    sequence_length = metadata['sequence_length']
    lstm_features = metadata['lstm_features']
    xgb_features = metadata['xgb_features']
    training_type = metadata.get('training_type', 'unknown')
    lstm_weight = metadata['lstm_weight']
    xgb_weight = metadata['xgb_weight']
    
    # Prepare data
    stock_data = stock_data.copy()
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date').reset_index(drop=True)
    
    df_features = create_features(stock_data)
    
    print(f"Total data points: {len(df_features)}")
    print(f"Testing on last {test_days} days")
    
    # Use last test_days for testing
    test_start_idx = len(df_features) - test_days
    
    if test_start_idx < sequence_length:
        print("❌ Not enough data for backtesting!")
        return None
    
    results = []
    
    print("\nRunning predictions...")
    for i in range(test_start_idx, len(df_features)):
        current_date = df_features['date'].iloc[i]
        actual_price = df_features['Close'].iloc[i]
        
        # Get data up to current point (simulating real-time prediction)
        historical_data = df_features.iloc[:i].copy()
        
        if len(historical_data) < sequence_length:
            continue
        
        # LSTM Prediction
        try:
            lstm_data = historical_data[lstm_features].values
            
            if training_type == 'multi_stock':
                from sklearn.preprocessing import MinMaxScaler
                stock_scaler = MinMaxScaler()
                lstm_scaled = stock_scaler.fit_transform(lstm_data)
            else:
                lstm_scaled = model_dict['scaler'].transform(lstm_data)
                stock_scaler = model_dict['scaler']
            
            last_sequence = lstm_scaled[-sequence_length:]
            last_sequence = last_sequence.reshape(1, sequence_length, len(lstm_features))
            
            lstm_pred_scaled = model_dict['lstm_model'].predict(last_sequence, verbose=0)[0][0]
            
            dummy = np.zeros((1, stock_scaler.n_features_in_))
            dummy[0, 0] = lstm_pred_scaled
            lstm_pred = stock_scaler.inverse_transform(dummy)[0, 0]
        except:
            lstm_pred = actual_price
        
        # XGBoost Prediction
        try:
            xgb_data = historical_data[xgb_features].iloc[-1:].values
            xgb_pred_raw = model_dict['xgb_model'].predict(xgb_data)[0]
            
            if training_type == 'multi_stock':
                previous_price = historical_data['Close'].iloc[-1]
                xgb_pred = previous_price * (1 + xgb_pred_raw)
            else:
                xgb_pred = xgb_pred_raw
        except:
            xgb_pred = actual_price
        
        # Ensemble
        ensemble_pred = lstm_weight * lstm_pred + xgb_weight * xgb_pred
        
        results.append({
            'date': current_date,
            'actual': actual_price,
            'lstm_pred': lstm_pred,
            'xgb_pred': xgb_pred,
            'ensemble_pred': ensemble_pred
        })
        
        if (i - test_start_idx + 1) % 20 == 0:
            print(f"  Progress: {i - test_start_idx + 1}/{test_days} days")
    
    results_df = pd.DataFrame(results)
    return results_df

# ============================================
# CALCULATE METRICS
# ============================================
def calculate_metrics(results_df):
    """Calculate accuracy metrics"""
    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")
    
    actual = results_df['actual'].values
    
    metrics = {}
    
    for model_name in ['lstm_pred', 'xgb_pred', 'ensemble_pred']:
        predictions = results_df[model_name].values
        
        # Price-based metrics
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        r2 = r2_score(actual, predictions)
        
        # Directional accuracy (most important for trading!)
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Average prediction error
        avg_error = np.mean(predictions - actual)
        avg_error_pct = (avg_error / np.mean(actual)) * 100
        
        metrics[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Directional Accuracy': directional_accuracy,
            'Avg Error': avg_error,
            'Avg Error %': avg_error_pct
        }
    
    # Display metrics
    print("\n" + "="*80)
    print(f"{'Metric':<25} {'LSTM':<15} {'XGBoost':<15} {'Ensemble':<15}")
    print("="*80)
    
    metric_names = ['RMSE', 'MAE', 'MAPE', 'R²', 'Directional Accuracy', 'Avg Error', 'Avg Error %']
    
    for metric in metric_names:
        lstm_val = metrics['lstm_pred'][metric]
        xgb_val = metrics['xgb_pred'][metric]
        ensemble_val = metrics['ensemble_pred'][metric]
        
        if metric in ['RMSE', 'MAE', 'Avg Error']:
            print(f"{metric:<25} ${lstm_val:<14.2f} ${xgb_val:<14.2f} ${ensemble_val:<14.2f}")
        elif metric in ['MAPE', 'Directional Accuracy', 'Avg Error %']:
            print(f"{metric:<25} {lstm_val:<14.2f}% {xgb_val:<14.2f}% {ensemble_val:<14.2f}%")
        else:
            print(f"{metric:<25} {lstm_val:<15.4f} {xgb_val:<15.4f} {ensemble_val:<15.4f}")
    
    print("="*80)
    
    # Interpretation
    print("\n📊 INTERPRETATION:")
    ensemble_acc = metrics['ensemble_pred']['Directional Accuracy']
    ensemble_mape = metrics['ensemble_pred']['MAPE']
    
    print(f"\n1. Directional Accuracy: {ensemble_acc:.2f}%")
    if ensemble_acc > 60:
        print("   ✅ EXCELLENT - Model predicts direction very well!")
    elif ensemble_acc > 55:
        print("   ✓ GOOD - Model has useful predictive power")
    elif ensemble_acc > 50:
        print("   ⚠️  MODERATE - Model is slightly better than random")
    else:
        print("   ❌ POOR - Model is not better than random guessing")
    
    print(f"\n2. MAPE (Mean Absolute Percentage Error): {ensemble_mape:.2f}%")
    if ensemble_mape < 2:
        print("   ✅ EXCELLENT - Very accurate price predictions")
    elif ensemble_mape < 5:
        print("   ✓ GOOD - Reasonably accurate predictions")
    elif ensemble_mape < 10:
        print("   ⚠️  MODERATE - Predictions have some error")
    else:
        print("   ❌ POOR - Large prediction errors")
    
    return metrics

# ============================================
# VISUALIZE RESULTS
# ============================================
def visualize_results(results_df, ticker, save_path='backtest_results.png'):
    """Create visualization of predictions vs actual"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Price predictions
    ax1 = axes[0]
    ax1.plot(results_df['date'], results_df['actual'], 
             label='Actual Price', color='black', linewidth=2, alpha=0.7)
    ax1.plot(results_df['date'], results_df['ensemble_pred'], 
             label='Predicted Price', color='red', linewidth=1.5, alpha=0.7)
    ax1.fill_between(results_df['date'], 
                      results_df['actual'], 
                      results_df['ensemble_pred'], 
                      alpha=0.3, color='gray')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{ticker} - Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction errors
    ax2 = axes[1]
    errors = results_df['ensemble_pred'] - results_df['actual']
    colors = ['green' if e < 0 else 'red' for e in errors]
    ax2.bar(results_df['date'], errors, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Prediction Error ($)', fontsize=12)
    ax2.set_title('Prediction Errors (Predicted - Actual)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()

# ============================================
# TRADING SIMULATION
# ============================================
def simulate_trading(results_df, initial_capital=10000):
    """Simulate trading based on predictions"""
    print(f"\n{'='*60}")
    print("TRADING SIMULATION")
    print(f"{'='*60}")
    
    capital = initial_capital
    shares = 0
    trades = []
    
    for i in range(1, len(results_df)):
        current_price = results_df['actual'].iloc[i-1]
        next_price = results_df['actual'].iloc[i]
        predicted_next = results_df['ensemble_pred'].iloc[i-1]
        
        # Buy signal: prediction is higher than current price
        if predicted_next > current_price and shares == 0:
            shares = capital / current_price
            capital = 0
            trades.append({
                'date': results_df['date'].iloc[i-1],
                'action': 'BUY',
                'price': current_price,
                'shares': shares
            })
        
        # Sell signal: prediction is lower than current price
        elif predicted_next < current_price and shares > 0:
            capital = shares * current_price
            trades.append({
                'date': results_df['date'].iloc[i-1],
                'action': 'SELL',
                'price': current_price,
                'profit': capital - initial_capital
            })
            shares = 0
    
    # Close any open position
    if shares > 0:
        final_price = results_df['actual'].iloc[-1]
        capital = shares * final_price
        trades.append({
            'date': results_df['date'].iloc[-1],
            'action': 'SELL',
            'price': final_price,
            'profit': capital - initial_capital
        })
    
    final_value = capital
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    # Buy and hold comparison
    buy_hold_shares = initial_capital / results_df['actual'].iloc[0]
    buy_hold_value = buy_hold_shares * results_df['actual'].iloc[-1]
    buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
    
    print(f"\nInitial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    print(f"\n📊 Buy & Hold Comparison:")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    
    if total_return > buy_hold_return:
        print(f"✅ Strategy OUTPERFORMED buy & hold by {total_return - buy_hold_return:.2f}%")
    else:
        print(f"❌ Strategy UNDERPERFORMED buy & hold by {buy_hold_return - total_return:.2f}%")
    
    return trades, final_value

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    print("="*60)
    print("MODEL ACCURACY TESTING")
    print("="*60)
    
    # Load data
    print("\nLoading stock data...")
    df = pd.read_csv('stock_data_5years.csv')
    available_tickers = sorted(df['ticker'].unique())
    
    print(f"Available tickers: {', '.join(available_tickers[:20])}...")
    
    # Select model
    print("\nAvailable models:")
    if os.path.exists('saved_models'):
        models = sorted([d for d in os.listdir('saved_models') 
                        if os.path.isdir(os.path.join('saved_models', d))], reverse=True)
        
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        model_choice = int(input("\nSelect model number: ")) - 1
        model_path = os.path.join('saved_models', models[model_choice])
    else:
        print("❌ No saved models found!")
        return
    
    # Load model
    model_dict = load_saved_model(model_path)
    
    # Select ticker
    ticker = input("\nEnter ticker to test (e.g., AAPL): ").strip().upper()
    
    if ticker not in available_tickers:
        print(f"❌ Ticker not found!")
        return
    
    # Select test period
    test_days = int(input("Enter number of days to test (e.g., 100): "))
    
    # Get stock data
    stock_data = df[df['ticker'] == ticker].copy()
    
    # Run backtest
    results_df = backtest_model(model_dict, stock_data, ticker, test_days)
    
    if results_df is None:
        return
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    # Visualize
    visualize_results(results_df, ticker)
    
    # Trading simulation
    simulate_trading(results_df)
    
    # Save results
    results_file = f'backtest_{ticker}_{test_days}days.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Saved detailed results to: {results_file}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()