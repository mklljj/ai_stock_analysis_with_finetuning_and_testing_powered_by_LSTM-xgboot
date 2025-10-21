import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def add_stocks_to_dataset(new_tickers, years=5, csv_file='stock_data_5years.csv'):
    """
    Add new stocks to existing dataset
    
    Parameters:
    - new_tickers: List of ticker symbols to add
    - years: Number of years of historical data
    - csv_file: Existing CSV file to append to
    """
    print(f"\n{'='*60}")
    print("ADDING NEW STOCKS TO DATASET")
    print(f"{'='*60}\n")
    
    # Load existing data
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        existing_tickers = set(existing_df['ticker'].unique())
        print(f"✓ Loaded existing dataset: {len(existing_tickers)} stocks")
    else:
        existing_df = pd.DataFrame()
        existing_tickers = set()
        print("⚠️  No existing dataset found, will create new one")
    
    # Filter out tickers that already exist
    new_tickers = [t for t in new_tickers if t not in existing_tickers]
    
    if not new_tickers:
        print("\n❌ All specified tickers already exist in the dataset!")
        return
    
    print(f"\nTickers to add: {len(new_tickers)}")
    print(f"List: {', '.join(new_tickers)}\n")
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    all_data = []
    failed_tickers = []
    
    print(f"Fetching data from {start_date.date()} to {end_date.date()}\n")
    
    for i, ticker in enumerate(new_tickers, 1):
        try:
            print(f"[{i}/{len(new_tickers)}] Fetching {ticker}...", end=' ')
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)
            
            if not df.empty:
                df['ticker'] = ticker
                df['date'] = df.index
                all_data.append(df)
                print(f"✓ ({len(df)} rows)")
            else:
                print(f"✗ No data")
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_tickers.append(ticker)
    
    # Combine new data
    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns to match existing format
        columns = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        new_df = new_df[columns]
        
        # Combine with existing data
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Remove duplicates (just in case)
        combined_df = combined_df.drop_duplicates(subset=['date', 'ticker'], keep='first')
        
        # Save
        combined_df.to_csv(csv_file, index=False)
        
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"✓ Added {len(new_tickers) - len(failed_tickers)} new stocks")
        print(f"✓ Total stocks in dataset: {combined_df['ticker'].nunique()}")
        print(f"✓ Total rows: {len(combined_df):,}")
        print(f"✓ Saved to: {csv_file}")
        
        if failed_tickers:
            print(f"\n⚠️  Failed to fetch: {', '.join(failed_tickers)}")
        
        # Also update parquet if it exists
        parquet_file = csv_file.replace('.csv', '.parquet')
        if os.path.exists(parquet_file) or len(combined_df) > 0:
            try:
                combined_df.to_parquet(parquet_file, compression='snappy')
                print(f"✓ Updated: {parquet_file}")
            except:
                print("⚠️  Parquet update skipped")
        
        return combined_df
    else:
        print("\n❌ No data was fetched successfully!")
        return None

def interactive_add_stocks():
    """Interactive mode to add stocks"""
    print("="*60)
    print("ADD STOCKS TO DATASET")
    print("="*60)
    
    print("\nCurrent dataset: stock_data_5years.csv")
    
    # Show current stocks
    if os.path.exists('stock_data_5years.csv'):
        df = pd.read_csv('stock_data_5years.csv')
        current_tickers = sorted(df['ticker'].unique())
        print(f"\nCurrent stocks ({len(current_tickers)}):")
        
        # Display in columns
        cols = 5
        for i in range(0, len(current_tickers), cols):
            row = current_tickers[i:i+cols]
            print("  " + "  ".join(f"{t:<10}" for t in row))
    
    print("\n" + "="*60)
    print("OPTIONS:")
    print("="*60)
    print("1. Add specific stocks (enter tickers)")
    print("2. Add all S&P 500 stocks")
    print("3. Add popular tech stocks")
    print("4. Add Dow Jones 30 stocks")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Manual entry
        print("\nEnter ticker symbols separated by commas")
        print("Example: ADP,NFLX,UBER,PYPL")
        ticker_input = input("\nTickers: ").strip().upper()
        new_tickers = [t.strip() for t in ticker_input.split(',')]
        
    elif choice == "2":
        # All S&P 500
        print("\nFetching S&P 500 list from Wikipedia...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_df = tables[0]
            new_tickers = sp500_df['Symbol'].tolist()
            new_tickers = [t.replace('.', '-') for t in new_tickers]
            print(f"✓ Found {len(new_tickers)} S&P 500 stocks")
        except Exception as e:
            print(f"❌ Error fetching S&P 500 list: {e}")
            print("Using backup list...")
            new_tickers = get_sp500_backup()
    
    elif choice == "3":
        # Popular tech stocks
        new_tickers = [
            'NFLX', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SQ', 'PYPL', 
            'SHOP', 'ROKU', 'DOCU', 'ZM', 'CRWD', 'SNOW', 'DDOG',
            'NET', 'TEAM', 'OKTA', 'SPLK', 'WDAY', 'NOW',
            'PANW', 'FTNT', 'CHKP', 'MDB', 'ESTC'
        ]
        print(f"\n✓ Selected {len(new_tickers)} tech stocks")
    
    elif choice == "4":
        # Dow Jones 30
        new_tickers = [
            'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS',
            'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD',
            'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM',
            'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW'
        ]
        print(f"\n✓ Selected {len(new_tickers)} Dow Jones stocks")
    
    else:
        print("❌ Invalid choice!")
        return
    
    # Confirm
    print(f"\nReady to add {len(new_tickers)} stocks")
    confirm = input("Proceed? (y/n): ").strip().lower()
    
    if confirm == 'y':
        add_stocks_to_dataset(new_tickers)
    else:
        print("❌ Cancelled")

def get_sp500_backup():
    """Backup S&P 500 list if Wikipedia fetch fails"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
        'PEP', 'AVGO', 'KO', 'COST', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT',
        'DHR', 'VZ', 'ADBE', 'DIS', 'NKE', 'CMCSA', 'TXN', 'NEE', 'PM', 'CRM',
        'UPS', 'RTX', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'HON', 'BA', 'AMGN',
        'LOW', 'SPGI', 'INTU', 'CAT', 'GE', 'AXP', 'BKNG', 'ISRG', 'TJX', 'ADP',
        'GILD', 'MMC', 'BLK', 'MDLZ', 'SYK', 'ADI', 'VRTX', 'CB', 'REGN', 'LMT',
        'AMT', 'C', 'PLD', 'SO', 'CI', 'MO', 'ZTS', 'BSX', 'DUK', 'EOG',
        'BMY', 'MMM', 'PNC', 'BDX', 'USB', 'TGT', 'CL', 'FI', 'CSX', 'APD',
        'NSC', 'WM', 'EQIX', 'DE', 'SHW', 'NOC', 'GD', 'AON', 'ITW', 'HUM',
        # Add more as needed...
    ]

if __name__ == "__main__":
    interactive_add_stocks()