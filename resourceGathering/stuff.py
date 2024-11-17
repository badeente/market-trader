import pandas as pd
df = pd.read_parquet('C:\\workspace\\market-trader\\backtest\\data\\test\\BTC-USDT.parquet\\BTC-USDT.parquet')
df.to_csv('C:\\workspace\\market-trader\\backtest\\data\\test\\BTC-USDT.parquet\\BTC-USDT.csv')