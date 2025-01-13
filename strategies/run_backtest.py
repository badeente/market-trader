import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from backtesting import Backtest, Strategy
from strategies.two_legged_pullback_strategy import TwoLeggedPullbackStrategy, DataFrameRow
from csv_loader_candlestickchart import CsvLoaderCandlestickChart

class TwoLeggedPullbackBacktest(Strategy):
    def init(self):
        self.strategy = TwoLeggedPullbackStrategy()

    def next(self):
        row = DataFrameRow(
            datetime=str(self.data.index[-1]),
            open=self.data.Open[-1],
            high=self.data.High[-1],
            low=self.data.Low[-1],
            close=self.data.Close[-1],
            volume=int(self.data.Volume[-1])
        )
        prediction = self.strategy.predict(row)
        
        if prediction.action == 'buy':
            self.buy()
        elif prediction.action == 'sell':
            self.sell()

def run_backtest(csv_file_path: str):
    loader = CsvLoaderCandlestickChart()
    df = loader.load_csv(csv_file_path)
    
    # Rename columns to match the required format
    df.columns = [col.capitalize() for col in df.columns]
    
    # Ensure the DataFrame has the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    bt = Backtest(df, TwoLeggedPullbackBacktest, cash=10000, commission=.002)
    stats = bt.run()
    bt.plot()
    
    return stats

if __name__ == "__main__":
    csv_file_path = "/home/andreas/workspace/market-trader/strategies/clean15.csv"
    stats = run_backtest(csv_file_path)
    print(stats)
