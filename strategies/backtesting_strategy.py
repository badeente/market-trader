from backtesting import Backtest, Strategy as BacktestStrategy
import pandas as pd
from typing import List
from strategies.strategy import Strategy, DataFrameRow, TradingAction
import plotly.graph_objects as go
from datetime import datetime, timedelta

class BacktestingStrategy(BacktestStrategy):
    # Define strategy as a class variable parameter
    strategy = None

    def init(self):
        # Called once at the start of the backtest
        self.current_position = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        try:
            # Create DataFrameRow from current candle data
            current_data = DataFrameRow(
                datetime=str(self.data.index[-1]),
                open=float(self.data.Open[-1]),
                high=float(self.data.High[-1]),
                low=float(self.data.Low[-1]),
                close=float(self.data.Close[-1]),
                volume=int(self.data.Volume[-1])
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
        except Exception as e:
            print(f"Error in next(): {str(e)}")
            raise

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
    print(f"Loading data from {csv_path}")
    
    try:
        # Load and prepare data
        data = pd.read_csv(csv_path)
        print(f"Loaded CSV with columns: {data.columns.tolist()}")
        print(f"First few rows:\n{data.head()}")
        
        # Create datetime index
        data.index = pd.RangeIndex(start=0, stop=len(data), step=1)
        data.index = pd.date_range(
            start=datetime.now().replace(hour=9, minute=30, second=0, microsecond=0),
            periods=len(data),
            freq='1min'
        )
        
        # Rename columns to match backtesting.py requirements
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        print(f"Processed data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Initialize and run backtest
        print("Initializing backtest...")
        bt = Backtest(
            data,
            BacktestingStrategy,
            cash=cash,
            commission=commission,
            exclusive_orders=True
        )
        
        # Pass strategy instance as a parameter
        print("Running backtest...")
        BacktestingStrategy.strategy = strategy  # Set strategy as class variable
        stats = bt.run()
        print("Backtest completed")
        
        # Generate plot
        print("Generating plots...")
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])

        # Save plot as HTML
        fig.write_html("backtest_results.html")
        print("Saved interactive plot to backtest_results.html")
        
        # Call plot function without any custom formatting
        bt.plot()
        print("Generated backtest plots")
        
        return stats, bt
        
    except Exception as e:
        print(f"Error in run_backtest: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        raise
