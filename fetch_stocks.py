import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_sp500_tickers():
    """Return S&P 500 ticker list"""
    # Top 50 S&P 500 stocks by market cap (you can expand this list)
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
        'PEP', 'AVGO', 'KO', 'COST', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT',
        'DHR', 'VZ', 'ADBE', 'DIS', 'NKE', 'CMCSA', 'TXN', 'NEE', 'PM', 'CRM',
        'UPS', 'RTX', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'HON', 'BA', 'AMGN'
    ]
    return tickers

def fetch_and_save_stocks(tickers, years=5, filename='stock_data_5years.csv'):
    """Fetch historical stock data and save to CSV"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    all_data = []
    failed_tickers = []
    
    print(f"Fetching data for {len(tickers)} stocks...")
    print(f"Date range: {start_date.date()} to {end_date.date()}\n")
    
    for i, ticker in enumerate(tickers, 1):
        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not df.empty:
                df['ticker'] = ticker
                df['date'] = df.index
                all_data.append(df)
                print(f"✓ [{i}/{len(tickers)}] {ticker} - {len(df)} rows")
            else:
                print(f"✗ [{i}/{len(tickers)}] {ticker} - No data")
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"✗ [{i}/{len(tickers)}] {ticker} - Error: {e}")
            failed_tickers.append(ticker)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns for better readability
        columns = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        combined_df = combined_df[columns]
        
        # Save as CSV
        combined_df.to_csv(filename, index=False)
        print(f"\n✅ Saved {len(combined_df):,} rows to '{filename}'")
        
        # Also save as Parquet (optional but recommended)
        try:
            parquet_filename = filename.replace('.csv', '.parquet')
            combined_df.to_parquet(parquet_filename, compression='snappy')
            print(f"✅ Also saved as '{parquet_filename}' (faster loading)")
        except:
            print("⚠️  Parquet save skipped (install pyarrow: pip install pyarrow)")
        
        if failed_tickers:
            print(f"\n⚠️  Failed tickers ({len(failed_tickers)}): {failed_tickers}")
        
        # Display summary
        print("\n📈 Data Summary:")
        print(f"Total rows: {len(combined_df):,}")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"Unique tickers: {combined_df['ticker'].nunique()}")
        print(f"\nFirst few rows:")
        print(combined_df.head(10))
        
        return combined_df
    else:
        print("❌ No data fetched!")
        return None

if __name__ == "__main__":
    # Get tickers
    print("Using predefined S&P 500 ticker list...\n")
    tickers = get_sp500_tickers()
    
    # Fetch and save data
    df = fetch_and_save_stocks(tickers, years=5)