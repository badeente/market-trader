import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeMetrics:
    def __init__(self):
        self.trades = []
        self.current_trade = None
        self.returns = []
        
    def record_action(self, timestamp, trade_params, equity):
        """Record trading action and update metrics with stop-loss and take-profit"""
        action = trade_params['action']
        if action != 0:  # Opening or switching position
            if self.current_trade is None:  # Opening new trade
                self.current_trade = {
                    'entry_time': timestamp,
                    'entry_equity': equity,
                    'position': 'long' if action == 1 else 'short',
                    'stop_loss': trade_params['long_stop_loss'] if action == 1 else trade_params['short_stop_loss'],
                    'take_profit': trade_params['long_take_profit'] if action == 1 else trade_params['short_take_profit']
                }
            else:  # Switching position
                self._close_trade(timestamp, equity)
                self.current_trade = {
                    'entry_time': timestamp,
                    'entry_equity': equity,
                    'position': 'long' if action == 1 else 'short',
                    'stop_loss': trade_params['long_stop_loss'] if action == 1 else trade_params['short_stop_loss'],
                    'take_profit': trade_params['long_take_profit'] if action == 1 else trade_params['short_take_profit']
                }
        elif action == 0 and self.current_trade is not None:  # Closing position
            self._close_trade(timestamp, equity)
    
    def _close_trade(self, exit_time, exit_equity):
        """Close current trade and calculate metrics"""
        if self.current_trade:
            self.current_trade['exit_time'] = exit_time
            self.current_trade['exit_equity'] = exit_equity
            self.current_trade['duration'] = exit_time - self.current_trade['entry_time']
            
            # Safely calculate return, avoiding division by zero
            entry_equity = self.current_trade['entry_equity']
            if entry_equity > 0:
                self.current_trade['return'] = (exit_equity - entry_equity) / entry_equity
            else:
                logger.warning(f"Invalid entry equity value: {entry_equity}. Setting return to 0.")
                self.current_trade['return'] = 0.0
            
            # Record if trade was closed due to stop-loss or take-profit
            if self.current_trade['return'] < 0:
                self.current_trade['exit_reason'] = 'stop_loss' if abs(self.current_trade['return']) >= self.current_trade['stop_loss']/100 else 'manual'
            else:
                self.current_trade['exit_reason'] = 'take_profit' if self.current_trade['return'] >= self.current_trade['take_profit']/100 else 'manual'
            
            self.trades.append(self.current_trade)
            self.returns.append(self.current_trade['return'])
            self.current_trade = None
    
    def calculate_metrics(self):
        """Calculate trading performance metrics including stop-loss and take-profit effectiveness"""
        if not self.trades:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'num_trades': 0,
                'avg_trade_duration': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'stop_loss_hits': 0,
                'take_profit_hits': 0,
                'manual_exits': 0
            }
        
        returns = np.array(self.returns)
        
        # Calculate average return and total return
        avg_return = np.mean(returns)
        total_return = np.prod(1 + returns) - 1  # Compound returns
        
        # Calculate Sharpe Ratio (assuming risk-free rate = 0)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return != 0 else 0
        
        # Calculate Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = avg_return / downside_std if downside_std != 0 else 0
        
        # Calculate other metrics
        durations = [trade['duration'] for trade in self.trades]
        avg_duration = np.mean(durations) if durations else 0
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        
        # Calculate stop-loss and take-profit effectiveness
        exit_reasons = [trade['exit_reason'] for trade in self.trades]
        stop_loss_hits = exit_reasons.count('stop_loss')
        take_profit_hits = exit_reasons.count('take_profit')
        manual_exits = exit_reasons.count('manual')
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'num_trades': len(self.trades),
            'avg_trade_duration': float(avg_duration),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'total_return': float(total_return),
            'stop_loss_hits': stop_loss_hits,
            'take_profit_hits': take_profit_hits,
            'manual_exits': manual_exits
        }
