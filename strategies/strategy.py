from enum import Enum
from dataclasses import dataclass
from typing import Protocol
from datetime import datetime

class TradingAction(str, Enum):
    BUY = 'buy'
    SELL = 'sell'
    NO_ACTION = 'noAction'

@dataclass
class PredictionResult:
    action: TradingAction
    stop_loss: float
    take_profit: float

@dataclass
class DataFrameRow:
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class Strategy(Protocol):
    def predict(self, row: DataFrameRow) -> PredictionResult:
        """
        Predicts trading action based on the current market data
        Args:
            row: Current market data row containing OHLCV values
        Returns:
            PredictionResult containing the trading action, stop loss, and take profit levels
        """
        pass
