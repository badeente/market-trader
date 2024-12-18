from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from strategies.strategy import Strategy, DataFrameRow, PredictionResult, TradingAction

@dataclass
class PullbackState:
    # Store both candle and its index
    first_leg_start: Optional[Tuple[DataFrameRow, int]] = None
    first_leg_end: Optional[Tuple[DataFrameRow, int]] = None
    bounce: Optional[Tuple[DataFrameRow, int]] = None
    second_leg_end: Optional[Tuple[DataFrameRow, int]] = None

class TwoLeggedPullbackStrategy(Strategy):
    def __init__(self):
        self.recent_candles: List[DataFrameRow] = []
        self.min_candles_needed = 4
        self.current_state = PullbackState()
        
        # Percentage threshold for identifying significant price movements
        self.movement_threshold = 0.003  # 0.3%
        self.bounce_threshold = 0.002    # 0.2%

    def predict(self, candle: DataFrameRow) -> PredictionResult:
        # Convert numpy types to Python native types
        candle = DataFrameRow(
            datetime=str(candle.datetime),
            open=float(candle.open),
            high=float(candle.high),
            low=float(candle.low),
            close=float(candle.close),
            volume=int(candle.volume)
        )
        
        # Add the new candle to our history
        self.recent_candles.append(candle)

        # Keep only the necessary amount of candles
        if len(self.recent_candles) > self.min_candles_needed:
            self.recent_candles.pop(0)
            # Reset state if we're removing a candle that's part of our pattern
            if (self.current_state.first_leg_start and 
                self.current_state.first_leg_start[1] >= len(self.recent_candles)):
                self.current_state = PullbackState()

        # Need minimum number of candles to make a prediction
        if len(self.recent_candles) < self.min_candles_needed:
            return self._no_action_result()

        return self._analyze_pullback_pattern()

    def _analyze_pullback_pattern(self) -> PredictionResult:
        # Reset state if we don't have enough recent downward movement
        if not self._has_downward_trend():
            self.current_state = PullbackState()
            return self._no_action_result()

        # Try to identify pattern components
        self._identify_pattern_components()

        # Check if we have a complete pattern
        if self._is_pattern_complete():
            entry_price = self.recent_candles[-1].close
            stop_loss = self._calculate_stop_loss()
            take_profit = self._calculate_take_profit(entry_price, stop_loss)

            # Reset state after generating signal
            self.current_state = PullbackState()

            return PredictionResult(
                action=TradingAction.BUY,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        return self._no_action_result()

    def _has_downward_trend(self) -> bool:
        lookback = min(3, len(self.recent_candles) - 1)
        current = self.recent_candles[-1]
        past = self.recent_candles[-1 - lookback]

        return (past.close - current.close) / past.close > self.movement_threshold

    def _identify_pattern_components(self) -> None:
        candles = self.recent_candles
        last_candle = candles[-1]

        # First leg identification
        if not self.current_state.first_leg_start:
            highest_idx = 0
            for i in range(1, len(candles) - 1):
                if candles[i].high > candles[highest_idx].high:
                    highest_idx = i
            self.current_state.first_leg_start = (candles[highest_idx], highest_idx)

        # First leg end (lowest point after first leg start)
        if self.current_state.first_leg_start and not self.current_state.first_leg_end:
            start_candle, start_idx = self.current_state.first_leg_start
            lowest_idx = start_idx
            
            for i in range(start_idx + 1, len(candles)):
                if candles[i].low < candles[lowest_idx].low:
                    lowest_idx = i
                    
            if ((start_candle.high - candles[lowest_idx].low) / 
                start_candle.high > self.movement_threshold):
                self.current_state.first_leg_end = (candles[lowest_idx], lowest_idx)

        # Bounce identification
        if self.current_state.first_leg_end and not self.current_state.bounce:
            end_candle, end_idx = self.current_state.first_leg_end
            highest_idx = end_idx
            
            for i in range(end_idx + 1, len(candles)):
                if candles[i].high > candles[highest_idx].high:
                    highest_idx = i
                    
            if ((candles[highest_idx].high - end_candle.low) / 
                end_candle.low > self.bounce_threshold):
                self.current_state.bounce = (candles[highest_idx], highest_idx)

        # Second leg end
        if (self.current_state.bounce and self.current_state.first_leg_end and 
            not self.current_state.second_leg_end):
            first_leg_end_candle = self.current_state.first_leg_end[0]
            if last_candle.low < first_leg_end_candle.low:
                self.current_state.second_leg_end = (last_candle, len(candles) - 1)

    def _is_pattern_complete(self) -> bool:
        return bool(
            self.current_state.first_leg_start and
            self.current_state.first_leg_end and
            self.current_state.bounce and
            self.current_state.second_leg_end
        )

    def _calculate_stop_loss(self) -> float:
        if not self._is_pattern_complete() or not self.current_state.second_leg_end:
            return 0

        # Set stop loss slightly below the second leg's low
        return self.current_state.second_leg_end[0].low * 0.998  # 0.2% below the low

    def _calculate_take_profit(self, entry_price: float, stop_loss: float) -> float:
        # Calculate take profit with 2:1 reward-to-risk ratio
        risk = entry_price - stop_loss
        return entry_price + (risk * 2)

    def _no_action_result(self) -> PredictionResult:
        return PredictionResult(
            action=TradingAction.NO_ACTION,
            stop_loss=0,
            take_profit=0
        )
