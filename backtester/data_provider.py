import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProvider:
    def __init__(self, csv_path):
        """
        Initialize DataProvider with CSV data
        
        Parameters:
        csv_path (str): Path to the CSV file containing trading data
        """
        try:
            self.df = pd.read_csv(csv_path)
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            self.counter = 0
            self.window_size = 200
            self.equity = 10000.0  # Starting equity
            self.market_position = 0  # 0: no position, 1: long, 2: short
            self.position_size = 1.0  # Standard position size
            self.transaction_fee = 0.001  # 0.1% transaction fee
            
            logger.info(f"Initialized DataProvider with {len(self.df)} records from {csv_path}")
            
        except Exception as e:
            logger.error(f"Error initializing DataProvider: {e}")
            raise
        
    def get_next_window(self):
        """
        Returns the next window of data
        
        Returns:
        pandas.DataFrame: DataFrame containing window_size entries or None if no more data
        """
        try:
            start_idx = self.counter
            end_idx = start_idx + self.window_size
            
            # Check if we have enough data left
            if end_idx > len(self.df):
                return None
                
            # Get the window
            window = self.df.iloc[start_idx:end_idx].copy()
            
            # Increment counter for next call
            self.counter += 1
            
            # Add technical indicators
            window = self._add_technical_indicators(window)
            
            return window
            
        except Exception as e:
            logger.error(f"Error getting next window: {e}")
            return None
    
    def _add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def calculate_equity(self, command):
        """
        Calculate equity change based on action and current market position
        
        Parameters:
        command (int): 0: close position, 1: go long, 2: go short
        
        Returns:
        float: Reward value based on action and position change
        """
        try:
            next_idx = self.counter + self.window_size
            if next_idx >= len(self.df):
                return 0
            
            current_price = self.df['close'].iloc[next_idx-1]
            next_price = self.df['close'].iloc[next_idx]
            
            # Calculate price change percentage
            price_change = (next_price - current_price) / current_price
            
            # Initialize reward
            reward = 0
            
            # Handle different scenarios
            if command == 0:  # Close position
                if self.market_position != 0:
                    # Add small penalty for closing position to prevent excessive trading
                    reward = -self.transaction_fee
                    self.market_position = 0
                else:
                    reward = 0  # Neutral reward for maintaining no position
                    
            elif command == 1:  # Long position
                if self.market_position == 0:  # Opening new long
                    reward = price_change - self.transaction_fee
                    self.market_position = 1
                elif self.market_position == 1:  # Maintaining long
                    reward = price_change
                else:  # Switching from short to long
                    reward = price_change - (2 * self.transaction_fee)  # Double fee for position switch
                    self.market_position = 1
                    
            elif command == 2:  # Short position
                if self.market_position == 0:  # Opening new short
                    reward = -price_change - self.transaction_fee
                    self.market_position = 2
                elif self.market_position == 2:  # Maintaining short
                    reward = -price_change
                else:  # Switching from long to short
                    reward = -price_change - (2 * self.transaction_fee)  # Double fee for position switch
                    self.market_position = 2
            
            # Update equity
            self.equity *= (1 + reward)
            
            # Scale reward for better learning
            scaled_reward = np.clip(reward * 100, -1, 1)  # Clip between -1 and 1
            
            logger.debug(f"Action: {command}, Position: {self.market_position}, "
                        f"Reward: {scaled_reward:.4f}, Equity: {self.equity:.2f}")
            
            return scaled_reward
            
        except Exception as e:
            logger.error(f"Error calculating equity: {e}")
            return 0
    
    def get_state(self):
        """
        Get current state information
        
        Returns:
        dict: Current state including position, equity, etc.
        """
        return {
            'market_position': self.market_position,
            'equity': self.equity,
            'counter': self.counter,
            'total_steps': len(self.df) - self.window_size
        }

if __name__ == "__main__":
    # Example usage
    try:
        provider = DataProvider("data/btc/5min/BTCUSDT.csv")
        
        # Get windows until None is returned
        window_count = 0
        while True:
            window = provider.get_next_window()
            if window is None:
                logger.info("No more data available")
                break
                
            logger.info(f"Window {window_count}:")
            logger.info(f"Start index: {provider.counter-1}")
            logger.info(f"End index: {provider.counter-1 + provider.window_size}")
            logger.info(f"Shape: {window.shape}")
            logger.info("-" * 50)
            window_count += 1
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
