import pandas as pd

def clean_timestamp(timestamp_str):
    # Remove milliseconds if they exist
    if '.' in timestamp_str:
        timestamp_str = timestamp_str.split('.')[0]
    return timestamp_str

def resample_to_5min(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Clean timestamps and convert to datetime
    df['open_time'] = df['open_time'].apply(clean_timestamp)
    df['open_time'] = pd.to_datetime(df['open_time'], format='%Y-%m-%d %H:%M:%S')
    
    # Set open_time as index
    df.set_index('open_time', inplace=True)
    
    # Resample to 5min intervals
    resampled = df.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_asset_volume': 'sum',
        'number_of_trades': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum'
    })
    
    # Reset index to make open_time a column again
    resampled.reset_index(inplace=True)
    
    # Save to CSV
    resampled.to_csv(output_file, index=False)
    print(f"Resampled data saved to {output_file}")

if __name__ == "__main__":
    input_file = r"C:\\workspace\\market-trader\\backtest\\data\\btc\\1min\\BTC-USDT.csv"
    output_file = r"C:\\workspace\\market-trader\\backtest\\data\\btc\\5min\\BTC-USDT-5min.csv"
    resample_to_5min(input_file, output_file)
