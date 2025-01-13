import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TrendAnalyzer:
    def __init__(self, data, price_column="close"):
        """
        Initialize with candlestick data.
        :param data: A pandas DataFrame with columns: 'datetime', 'open', 'high', 'low', 'close', 'volume'.
        :param price_column: The column to analyze (default: 'close').
        """
        required_columns = {"datetime", "open", "high", "low", "close", "volume"}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Data must contain the following columns: {required_columns}")
        self.data = data
        self.prices = data[price_column]

    def calculate_ema(self, window):
        """
        Calculate Exponential Moving Average (EMA).
        :param window: The EMA window size.
        :return: A pandas Series of EMA values.
        """
        return self.prices.ewm(span=window, adjust=False).mean()

    def linear_regression_slope(self, data, price_column="close"):
        """
        Calculate the slope of a linear regression fitted to the price data.
        :return: The slope of the regression line.
        """
        self.prices = data[price_column]
        x = np.arange(len(self.prices)).reshape(-1, 1)
        y = self.prices.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_[0][0]  # The slope

    def determine_trend(self, ema_window=10, tolerance=0.01):
        """
        Determine the trend based on EMA slope and linear regression.
        :param ema_window: EMA window size.
        :param tolerance: Flat slope tolerance for a 'Ranging' trend.
        :return: The trend ('Uptrend', 'Downtrend', 'Ranging').
        """
        lr_slope = self.linear_regression_slope()

        # Combine EMA slope and Linear Regression slope to classify trend
        if lr_slope > 0:
            return "Uptrend"
        elif lr_slope < 0:
            return "Downtrend"
        else:
            return "Ranging"

# Example Usage
if __name__ == "__main__":
    # Example candlestick DataFrame
    data = pd.DataFrame({
        "datetime": pd.date_range(start="2023-01-01", periods=200, freq="D"),
        "open": np.random.uniform(100, 110, 200),
        "high": np.random.uniform(105, 115, 200),
        "low": np.random.uniform(95, 105, 200),
        "close": np.random.uniform(100, 110, 200),
        "volume": np.random.randint(1000, 5000, 200),
    })

    # Take the last 100 candlesticks
    last_100 = data.iloc[-100:]

    # Initialize the analyzer
    analyzer = TrendAnalyzer(last_100)

    # Determine the trend
    trend = analyzer.determine_trend(ema_window=10)
    print(f"Detected Trend: {trend}")
