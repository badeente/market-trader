from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Literal
import numpy as np

@dataclass
class CandlestickData:
    timestamp: int
    open: float
    close: float
    high: float
    low: float
    volume: float

class CandlestickPattern(str, Enum):
    BULLISH_ENGULFING = 'bullish_engulfing'
    BEARISH_ENGULFING = 'bearish_engulfing'
    DOJI = 'doji'
    HAMMER = 'hammer'
    SHOOTING_STAR = 'shooting_star'
    NONE = 'none'

@dataclass
class TradingDecision:
    decision: Literal['buy', 'sell', 'noAction']
    stop_loss: float
    take_profit: float

class AdvancedCandlestickPatternAnalyzer:
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    def analyze_candlestick_pattern(
        self,
        current_candle: CandlestickData,
        previous_candles: List[CandlestickData]
    ) -> TradingDecision:
        if len(previous_candles) < max(self.RSI_PERIOD, self.MACD_SLOW):
            return TradingDecision(decision='noAction', stop_loss=0, take_profit=0)

        pattern = self._detect_candlestick_pattern(current_candle, previous_candles[0])
        rsi, macd = self._calculate_technical_indicators(current_candle, previous_candles)
        return self._make_trading_decision(current_candle, pattern, rsi, macd)

    def _detect_candlestick_pattern(
        self,
        current: CandlestickData,
        previous: CandlestickData
    ) -> CandlestickPattern:
        curr_open, curr_close = current.open, current.close
        curr_high, curr_low = current.high, current.low
        prev_open, prev_close = previous.open, previous.close

        body_length = abs(curr_close - curr_open)
        total_length = curr_high - curr_low
        upper_shadow = curr_high - max(curr_open, curr_close)
        lower_shadow = min(curr_open, curr_close) - curr_low

        # Doji pattern
        if body_length <= 0.1 * total_length:
            return CandlestickPattern.DOJI

        # Bullish engulfing
        if (curr_close > curr_open and  # Current candle is bullish
            prev_close < prev_open and   # Previous candle was bearish
            curr_open < prev_close and   # Current opens below previous close
            curr_close > prev_open):     # Current closes above previous open
            return CandlestickPattern.BULLISH_ENGULFING

        # Bearish engulfing
        if (curr_close < curr_open and   # Current candle is bearish
            prev_close > prev_open and    # Previous candle was bullish
            curr_open > prev_close and    # Current opens above previous close
            curr_close < prev_open):      # Current closes below previous open
            return CandlestickPattern.BEARISH_ENGULFING

        # Hammer
        if (curr_close > curr_open and      # Bullish candle
            lower_shadow > 2 * body_length and  # Long lower shadow
            upper_shadow < 0.1 * total_length): # Small upper shadow
            return CandlestickPattern.HAMMER

        # Shooting star
        if (curr_close < curr_open and      # Bearish candle
            upper_shadow > 2 * body_length and  # Long upper shadow
            lower_shadow < 0.1 * total_length): # Small lower shadow
            return CandlestickPattern.SHOOTING_STAR

        return CandlestickPattern.NONE

    def _calculate_technical_indicators(
        self,
        current: CandlestickData,
        previous_candles: List[CandlestickData]
    ) -> Tuple[float, float]:
        rsi = self._calculate_rsi([current] + previous_candles)
        macd = self._calculate_macd([current] + previous_candles)
        return rsi, macd

    def _calculate_rsi(self, candles: List[CandlestickData]) -> float:
        if len(candles) < self.RSI_PERIOD + 1:
            return 50

        gains = 0
        losses = 0

        # Calculate initial average gain and loss
        for i in range(1, self.RSI_PERIOD + 1):
            difference = candles[i-1].close - candles[i].close
            if difference >= 0:
                gains += difference
            else:
                losses -= difference

        avg_gain = gains / self.RSI_PERIOD
        avg_loss = losses / self.RSI_PERIOD

        # Calculate subsequent values using smoothing
        for i in range(self.RSI_PERIOD + 1, len(candles)):
            difference = candles[i-1].close - candles[i].close
            if difference >= 0:
                avg_gain = (avg_gain * (self.RSI_PERIOD - 1) + difference) / self.RSI_PERIOD
                avg_loss = (avg_loss * (self.RSI_PERIOD - 1)) / self.RSI_PERIOD
            else:
                avg_gain = (avg_gain * (self.RSI_PERIOD - 1)) / self.RSI_PERIOD
                avg_loss = (avg_loss * (self.RSI_PERIOD - 1) - difference) / self.RSI_PERIOD

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, candles: List[CandlestickData]) -> float:
        if len(candles) < self.MACD_SLOW:
            return 0

        prices = [c.close for c in candles][::-1]  # Reverse list for correct order
        fast_ema = self._calculate_ema(prices, self.MACD_FAST)
        slow_ema = self._calculate_ema(prices, self.MACD_SLOW)
        macd_line = fast_ema - slow_ema
        signal_line = self._calculate_ema(
            [0] * (self.MACD_SLOW - 1) + [macd_line],
            self.MACD_SIGNAL
        )

        return macd_line - signal_line  # MACD histogram

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        multiplier = 2 / (period + 1)
        ema = prices[0]

        for i in range(1, len(prices)):
            ema = (prices[i] - ema) * multiplier + ema

        return ema

    def _make_trading_decision(
        self,
        candlestick: CandlestickData,
        pattern: CandlestickPattern,
        rsi: float,
        macd: float
    ) -> TradingDecision:
        high, low = candlestick.high, candlestick.low
        volatility_factor = 0.02  # 2% for stop loss and take profit

        # Strong buy signals
        if ((pattern in [CandlestickPattern.BULLISH_ENGULFING, CandlestickPattern.HAMMER]) and 
            rsi < self.RSI_OVERSOLD and 
            macd > 0):
            return TradingDecision(
                decision='buy',
                stop_loss=low * (1 - volatility_factor),
                take_profit=high * (1 + volatility_factor * 2)
            )

        # Strong sell signals
        if ((pattern in [CandlestickPattern.BEARISH_ENGULFING, CandlestickPattern.SHOOTING_STAR]) and 
            rsi > self.RSI_OVERBOUGHT and 
            macd < 0):
            return TradingDecision(
                decision='sell',
                stop_loss=high * (1 + volatility_factor),
                take_profit=low * (1 - volatility_factor * 2)
            )

        # Moderate buy signals
        if ((pattern in [CandlestickPattern.BULLISH_ENGULFING, CandlestickPattern.HAMMER]) and 
            (rsi < 45 or macd > 0)):
            return TradingDecision(
                decision='buy',
                stop_loss=low * (1 - volatility_factor * 0.75),
                take_profit=high * (1 + volatility_factor * 1.5)
            )

        # Moderate sell signals
        if ((pattern in [CandlestickPattern.BEARISH_ENGULFING, CandlestickPattern.SHOOTING_STAR]) and 
            (rsi > 55 or macd < 0)):
            return TradingDecision(
                decision='sell',
                stop_loss=high * (1 + volatility_factor * 0.75),
                take_profit=low * (1 - volatility_factor * 1.5)
            )

        # No clear signal
        return TradingDecision(
            decision='noAction',
            stop_loss=0,
            take_profit=0
        )
