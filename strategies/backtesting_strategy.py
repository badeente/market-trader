from backtesting import Backtest, Strategy as BacktestStrategy
import pandas as pd
from typing import List
from strategies.strategy import Strategy, DataFrameRow, TradingAction
import pandas as pd

class BacktestingStrategy(BacktestStrategy):
    # Define strategy as a class variable parameter
    strategy = None

    def init(self):
        # Called once at the start of the backtest
        self.current_position = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        # Create DataFrameRow from current candle data
        current_data = DataFrameRow(
            datetime=str(self.data.index[-1]),
            open=self.data.Open[-1],
            high=self.data.High[-1],
            low=self.data.Low[-1],
            close=self.data.Close[-1],
            volume=self.data.Volume[-1]
        )

        # Get prediction from strategy
        prediction = self.strategy.predict(current_data)

        # Handle existing position
        if self.position:
            # Check if stop loss or take profit hit
            if self.stop_loss and self.data.Low[-1] <= self.stop_loss:
                self.position.close()
                self.current_position = None
            elif self.take_profit and self.data.High[-1] >= self.take_profit:
                self.position.close()
                self.current_position = None

        # Handle new signals
        if not self.position:
            if prediction.action == TradingAction.BUY:
                self.buy()
                self.current_position = 'long'
                self.stop_loss = prediction.stop_loss
                self.take_profit = prediction.take_profit
            elif prediction.action == TradingAction.SELL:
                self.sell()
                self.current_position = 'short'
                self.stop_loss = prediction.stop_loss
                self.take_profit = prediction.take_profit

def run_backtest(csv_path: str, strategy: Strategy, cash: float = 10000, commission: float = 0.002):
    """
    Run a backtest using the provided strategy and data
    
    Args:
        csv_path: Path to CSV file containing OHLCV data
        strategy: Strategy implementation to use for predictions
        cash: Initial cash amount for backtest
        commission: Commission rate for trades
    
    Returns:
        Backtest results
    """
    # Load and prepare data
    data = pd.read_csv(csv_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    
    # Rename columns to match backtesting.py requirements
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Initialize and run backtest
    bt = Backtest(
        data,
        BacktestingStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True
    )
    
    # Pass strategy instance as a parameter
    stats = bt.run(strategy=strategy)
    return stats, bt
