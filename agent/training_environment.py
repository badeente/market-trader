import os
import time
import random
import logging
import numpy as np
from collections import deque
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from model import TradingAgent
from metrics import TradeMetrics
from early_stopping import EarlyStopping
from screenshot_manager import ScreenshotManager
from performance_profiler import PerformanceProfiler
from backtester.data_provider import DataProvider
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
INITIAL_EQUITY = 10000.0
DEFAULT_RISK_PER_TRADE = 0.001  # Changed from 0.0001 to 0.001 (0.1% risk per trade)
VALIDATION_FREQUENCY = 5  # Validate every N batches
VALIDATION_SUBSET_SIZE = 0.2  # Use 20% of validation data
VALIDATION_EARLY_STOP_THRESHOLD = 1.5  # Stop validation if loss is 1.5x worse than best
MIN_STOP_LOSS_PCT = 0.001  # Minimum 0.1% stop loss
DEFAULT_STOP_LOSS_PCT = 0.02  # Default 2% stop loss if none provided

@contextmanager
def dummy_profiler(*args, **kwargs):
    """Dummy context manager when profiling is disabled"""
    yield

def get_safe_value(params, key, default=DEFAULT_STOP_LOSS_PCT):
    """Safely get a value from trade parameters"""
    try:
        value = params.get(key)
        if value is None or np.isnan(value):
            logger.warning(f"Invalid {key}: {value}, using default {default}")
            return default
        return value
    except Exception as e:
        logger.error(f"Error getting {key}: {e}")
        return default

class Position:
    def __init__(self, position_type, entry_price, stop_loss_pct, max_risk_per_trade=DEFAULT_RISK_PER_TRADE):
        self.type = position_type  # 0=long, 1=short
        self.entry_price = entry_price
        
        try:
            # Ensure stop_loss_pct is a valid float
            stop_loss_pct = float(stop_loss_pct)
            
            # If percentage is given as whole number (e.g. 2 for 2%), convert to decimal
            if stop_loss_pct > 1:
                stop_loss_pct = stop_loss_pct / 100.0
            
            # Ensure minimum stop loss percentage
            stop_loss_pct = max(stop_loss_pct, MIN_STOP_LOSS_PCT)
            
            # Enforce maximum risk per trade
            if stop_loss_pct > max_risk_per_trade:
                #logger.warning(f"Stop loss {stop_loss_pct*100:.2f}% exceeds maximum risk {max_risk_per_trade*100:.2f}%. Using maximum allowed risk.")
                stop_loss_pct = max_risk_per_trade
            
            # If stop loss is somehow invalid, use default
            if stop_loss_pct <= 0 or np.isnan(stop_loss_pct):
                #logger.warning(f"Invalid stop loss {stop_loss_pct}, using default {DEFAULT_STOP_LOSS_PCT}")
                stop_loss_pct = DEFAULT_STOP_LOSS_PCT
            
            # Set stop loss based on input percentage
            self.stop_loss_price = entry_price * (1 - stop_loss_pct) if position_type == 0 else entry_price * (1 + stop_loss_pct)
            
            # Force take profit to be 2x the stop loss
            actual_take_profit_pct = stop_loss_pct * 2
            self.take_profit_price = entry_price * (1 + actual_take_profit_pct) if position_type == 0 else entry_price * (1 - actual_take_profit_pct)
            
            logger.info(f"Position created: Type={'Long' if position_type==0 else 'Short'}, "
                       f"Entry={entry_price:.2f}, SL={self.stop_loss_price:.2f} ({stop_loss_pct*100:.2f}%), "
                       f"TP={self.take_profit_price:.2f} ({actual_take_profit_pct*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error setting position parameters: {e}")
            # Use default values if there's an error
            stop_loss_pct = DEFAULT_STOP_LOSS_PCT
            self.stop_loss_price = entry_price * (1 - stop_loss_pct) if position_type == 0 else entry_price * (1 + stop_loss_pct)
            actual_take_profit_pct = stop_loss_pct * 2
            self.take_profit_price = entry_price * (1 + actual_take_profit_pct) if position_type == 0 else entry_price * (1 - actual_take_profit_pct)
            
        self.entry_time = time.time()
    
    def check_exit_conditions(self, current_price):
        """Check if position should be closed based on stop-loss or take-profit"""
        if self.type == 0:  # Long position
            if current_price <= self.stop_loss_price:
                return True, 'stop_loss'
            if current_price >= self.take_profit_price:
                return True, 'take_profit'
        else:  # Short position
            if current_price >= self.stop_loss_price:
                return True, 'stop_loss'
            if current_price <= self.take_profit_price:
                return True, 'take_profit'
        return False, None

class TrainingEnvironment:
    def __init__(self, train_data_path, val_data_path, model_save_dir, batch_size=16, memory_size=10000,
                 risk_per_trade=DEFAULT_RISK_PER_TRADE, screenshot_cache_size=1000,
                 validation_frequency=VALIDATION_FREQUENCY, validation_subset_size=VALIDATION_SUBSET_SIZE,
                 validation_early_stop_threshold=VALIDATION_EARLY_STOP_THRESHOLD, use_profiler=False):
        """Initialize the training environment with configurable parameters"""
        logger.info(f"Initializing training environment...")
        
        # Initialize performance profiler with log directory if enabled
        self.log_dir = os.path.join(model_save_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.use_profiler = use_profiler
        if use_profiler:
            self.profiler = PerformanceProfiler(log_dir=self.log_dir)
            self.profile = self.profiler.profile
        else:
            self.profiler = None
            self.profile = dummy_profiler
        
        logger.info(f"Training data path: {train_data_path}")
        logger.info(f"Validation data path: {val_data_path}")
        logger.info(f"Model save directory: {model_save_dir}")
        logger.info(f"Risk per trade: {risk_per_trade*100}%")
        logger.info(f"Screenshot cache size: {screenshot_cache_size}")
        logger.info(f"Validation frequency: Every {validation_frequency} batches")
        logger.info(f"Validation subset size: {validation_subset_size*100}%")
        logger.info(f"Performance profiling: {'Enabled' if use_profiler else 'Disabled'}")
        
        self.risk_per_trade = risk_per_trade
        self.agent = TradingAgent(max_risk_per_trade=risk_per_trade)
        self.train_data_provider = DataProvider(train_data_path)
        self.val_data_provider = DataProvider(val_data_path)
        self.model_save_dir = model_save_dir
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.episode_rewards = []
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.early_stopping = EarlyStopping(patience=5)
        
        # Validation parameters
        self.validation_frequency = validation_frequency
        self.validation_subset_size = validation_subset_size
        self.validation_early_stop_threshold = validation_early_stop_threshold
        self.batch_count = 0
        
        # Initialize screenshot manager
        self.screenshot_manager = ScreenshotManager(max_cache_size=screenshot_cache_size)
        
        # Add screenshot counters
        self.total_screenshots_seen = 0
        self.training_screenshots = 0
        self.validation_screenshots = 0
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_metrics = None
        
        # Episode metrics for plotting
        self.episode_train_losses = []
        self.episode_val_losses = []
        self.episode_equity_curves = []
        self.best_equity_curve = []
        self.best_episode = 0
        
        # Reset initial equity
        self.initial_equity = INITIAL_EQUITY
        self.train_data_provider.equity = INITIAL_EQUITY
        self.val_data_provider.equity = INITIAL_EQUITY
        logger.info(f"Initial equity set to: {self.initial_equity}")
        
        # Batch accumulation
        self.current_batch_screenshots = []
        self.current_batch_actions = []
        
        logger.info("Training environment initialized successfully")
    
    def reset_equity(self):
        """Reset equity to initial value for training only and select new random week"""
        # Reset to new random week
        initial_window = self.train_data_provider.reset_to_random_week()
        if initial_window is None:
            logger.error("Failed to reset to random week")
            return INITIAL_EQUITY
            
        state = self.train_data_provider.get_state()
        logger.info(f"Reset training equity to {INITIAL_EQUITY} and selected new week "
                   f"(indices {state['week_start']} to {state['week_end']})")
        return INITIAL_EQUITY
    
    def get_latest_screenshots(self, data_provider):
        """Get the latest market data as a PNG screenshot"""
        with self.profile('get_screenshots'):
            try:
                window = data_provider.get_next_window()
                if window is None:
                    return None, None, None
                    
                png_data = self.screenshot_manager.get_screenshot(window, num_entries=200)
                timestamp = window.index[-1] if hasattr(window, 'index') else len(window) - 1
                current_price = window.iloc[-1]['close'] if hasattr(window, 'close') else None
                
                # Increment screenshot counters
                self.total_screenshots_seen += 1
                if data_provider == self.train_data_provider:
                    self.training_screenshots += 1
                elif data_provider == self.val_data_provider:
                    self.validation_screenshots += 1
                
                if random.random() < 0.01:
                    cache_info = self.screenshot_manager.get_cache_info()
                    logger.debug(f"Screenshot cache info: {cache_info}")
                    
                return png_data, timestamp, current_price
            except Exception as e:
                logger.error(f"Error getting screenshots: {e}")
                return None, None, None
    
    def accumulate_batch(self, screenshot, action):
        """Accumulate samples until we have a full batch"""
        with self.profile('batch_accumulation'):
            self.current_batch_screenshots.append(screenshot)
            self.current_batch_actions.append(action)
            
            if len(self.current_batch_screenshots) >= self.batch_size:
                batch_screenshots = self.current_batch_screenshots
                batch_actions = self.current_batch_actions
                self.current_batch_screenshots = []
                self.current_batch_actions = []
                return batch_screenshots, batch_actions
            return None, None
    
    def should_validate(self):
        """Determine if validation should be performed"""
        self.batch_count += 1
        return self.batch_count % self.validation_frequency == 0
    
    def train_episode(self, episode_num, num_steps=100):
        """Train for one episode with validation"""
        with self.profile('train_episode'):
            episode_rewards = []
            train_losses = []
            val_losses = []
            equity_curve = []
            current_equity = self.reset_equity()  # This now also selects a new random week
            metrics = TradeMetrics()
            current_position = None  # Track current position
            
            # Get initial state to log week information
            state = self.train_data_provider.get_state()
            logger.info(f"\nEpisode {episode_num + 1} - Starting training with week "
                       f"from index {state['week_start']} to {state['week_end']}")
            logger.info(f"Episode {episode_num + 1} - Starting equity: {current_equity}")
            
            try:
                for step in range(num_steps):
                    screenshot_data, timestamp, current_price = self.get_latest_screenshots(self.train_data_provider)
                    if screenshot_data is None or current_price is None:
                        logger.info(f"Episode {episode_num + 1} - End of week reached at step {step}")
                        break
                    
                    # Check if current position should be closed
                    if current_position is not None:
                        should_close, reason = current_position.check_exit_conditions(current_price)
                        if should_close:
                            # Store position type before clearing position
                            position_type = current_position.type
                            
                            # Calculate reward based on exit reason
                            if reason == 'stop_loss':
                                reward = -current_position.stop_loss_price / current_position.entry_price + 1
                            else:  # take_profit
                                reward = current_position.take_profit_price / current_position.entry_price - 1
                            
                            if current_position.type == 1:  # Short position
                                reward = -reward
                            
                            current_equity *= (1 + reward)
                            # Pass proper trade_params with exit reason
                            metrics.record_action(timestamp, {'exit_reason': reason}, current_equity)
                            current_position = None
                            episode_rewards.append(reward)
                            
                            # Add to memory for training using stored position_type
                            self.memory.append((screenshot_data, position_type, reward))
                    
                    # Only look for new position if we don't have one
                    if current_position is None:
                        try:
                            trade_params = self.agent.predict(screenshot_data)
                            logger.debug(f"Trade params received: {trade_params}")
                            
                            if trade_params.get('should_trade', False):
                                action = get_safe_value(trade_params, 'action', 0)  # Default to long if action is invalid
                                
                                # Get stop loss with defensive programming
                                if action == 0:
                                    stop_loss = get_safe_value(trade_params, 'long_stop_loss')
                                else:
                                    stop_loss = get_safe_value(trade_params, 'short_stop_loss')
                                
                                logger.info(f"Creating position: Action={action}, Stop Loss={stop_loss}%")
                                
                                # Create new position with error handling and pass max risk per trade
                                current_position = Position(action, current_price, stop_loss, max_risk_per_trade=self.risk_per_trade)
                                metrics.record_action(timestamp, trade_params, current_equity)
                                
                                # Add to memory for training
                                self.memory.append((screenshot_data, action, 0))  # Initial reward is 0
                                
                        except Exception as e:
                            logger.error(f"Error processing trade: {e}")
                            logger.error(f"Trade params: {trade_params if 'trade_params' in locals() else 'Not available'}")
                            current_position = None  # Ensure position is None if creation failed
                    
                    if current_equity <= 0:
                        logger.warning(f"Episode {episode_num + 1} - Training equity reached {current_equity:.2f} at step {step}. Ending episode early.")
                        break
                    
                    equity_curve.append(current_equity)
                    
                    if len(self.memory) >= self.batch_size:
                        batch_data = random.sample(self.memory, self.batch_size)
                        for screenshot, action, _ in batch_data:
                            batch_screenshots, batch_actions = self.accumulate_batch(screenshot, action)
                            if batch_screenshots is not None:
                                for s, a in zip(batch_screenshots, batch_actions):
                                    loss, is_best = self.agent.train_step(s, a)
                                    train_losses.append(loss)
                                    
                                    if is_best:
                                        self.best_model_state = self.agent.model.state_dict().copy()
                                        self.best_val_loss = loss
                                        self.best_equity_curve = equity_curve.copy()
                                        self.best_episode = episode_num
                        
                        if self.should_validate():
                            val_loss, val_metrics = self.validate()
                            if val_loss is not None:
                                val_losses.append(val_loss)
                                
                                if val_loss < self.best_val_loss:
                                    self.best_val_loss = val_loss
                                    self.best_model_state = self.agent.model.state_dict().copy()
                                    self.best_equity_curve = equity_curve.copy()
                                    self.best_episode = episode_num
                                    self.best_metrics = val_metrics
                                    self.save_best_model()
                            
                                if self.early_stopping(val_loss):
                                    logger.info(f"Episode {episode_num + 1} - Early stopping triggered")
                                    break
                    
                    if step % 10 == 0:
                        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                        avg_train_loss = np.mean(train_losses[-10:]) if train_losses else 0.0
                        position_str = "None"
                        if current_position is not None:
                            position_str = "Long" if current_position.type == 0 else "Short"
                            position_str += f" (SL: {current_position.stop_loss_price:.2f}, TP: {current_position.take_profit_price:.2f})"
                        
                        logger.info(f"Episode {episode_num + 1} - Step {step}/{num_steps} - "
                                  f"Position: {position_str} - "
                                  f"Price: {current_price:.2f} - "
                                  f"Avg Reward: {avg_reward:.4f} - "
                                  f"Train Loss: {avg_train_loss:.4f} - "
                                  f"Current Equity: {current_equity:.2f}")
                    
                    time.sleep(0.1)
                
                self.episode_equity_curves.append(equity_curve)
                
                if episode_rewards:
                    avg_reward = np.mean(episode_rewards)
                    avg_train_loss = np.mean(train_losses) if train_losses else 0.0
                    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
                    
                    self.episode_rewards.append(avg_reward)
                    self.episode_train_losses.append(avg_train_loss)
                    self.episode_val_losses.append(avg_val_loss)
                    
                    logger.info(f"Episode {episode_num + 1} ended with equity: {current_equity:.2f}")
                    
                    # Save performance statistics at the end of each episode
                    if self.use_profiler:
                        self.profiler.save_statistics(f'performance_stats_episode_{episode_num + 1}.txt')
                    
                    return avg_reward, avg_train_loss, avg_val_loss, current_equity
                
                return 0.0, 0.0, 0.0, current_equity
                
            except Exception as e:
                logger.error(f"Error during episode {episode_num + 1}: {e}")
                return 0.0, 0.0, 0.0, current_equity
            finally:
                # Save final performance statistics for the episode
                if self.use_profiler:
                    self.profiler.save_statistics(f'performance_stats_episode_{episode_num + 1}.txt')
                    self.profiler.reset()  # Reset statistics for next episode
    
    def validate(self):
        """Validate the model using the validation data provider"""
        with self.profile('validation'):
            val_losses = []
            metrics = TradeMetrics()
            self.agent.model.eval()
            
            # Reset validation provider to new random week
            initial_window = self.val_data_provider.reset_to_random_week()
            if initial_window is None:
                logger.error("Failed to reset validation to random week")
                return None, None
                
            current_equity = self.val_data_provider.equity
            current_position = None
            
            # Get initial state to log week information
            state = self.val_data_provider.get_state()
            logger.info(f"Starting validation with week from index {state['week_start']} to {state['week_end']}")
            
            try:
                with torch.no_grad():
                    validation_steps = 0
                    max_validation_steps = int(1 / self.validation_subset_size)
                    
                    while validation_steps < max_validation_steps:
                        screenshot, timestamp, current_price = self.get_latest_screenshots(self.val_data_provider)
                        if screenshot is None or current_price is None:
                            logger.info("End of validation week reached")
                            break
                        
                        # Check if current position should be closed
                        if current_position is not None:
                            should_close, reason = current_position.check_exit_conditions(current_price)
                            if should_close:
                                # Store position type before clearing position
                                position_type = current_position.type
                                
                                if reason == 'stop_loss':
                                    reward = -current_position.stop_loss_price / current_position.entry_price + 1
                                else:  # take_profit
                                    reward = current_position.take_profit_price / current_position.entry_price - 1
                                
                                if current_position.type == 1:  # Short position
                                    reward = -reward
                                
                                current_equity *= (1 + reward)
                                # Pass proper trade_params with exit reason
                                metrics.record_action(timestamp, {'exit_reason': reason}, current_equity)
                                current_position = None
                        
                        # Only look for new position if we don't have one
                        if current_position is None:
                            try:
                                trade_params = self.agent.predict(screenshot)
                                logger.debug(f"Trade params received: {trade_params}")
                                
                                if trade_params.get('should_trade', False):
                                    action = get_safe_value(trade_params, 'action', 0)  # Default to long if action is invalid
                                    
                                    # Get stop loss with defensive programming
                                    if action == 0:
                                        stop_loss = get_safe_value(trade_params, 'long_stop_loss')
                                    else:
                                        stop_loss = get_safe_value(trade_params, 'short_stop_loss')
                                    
                                    logger.info(f"Creating position: Action={action}, Stop Loss={stop_loss}%")
                                    
                                    # Create new position with error handling and pass max risk per trade
                                    current_position = Position(action, current_price, stop_loss, max_risk_per_trade=self.risk_per_trade)
                                    metrics.record_action(timestamp, trade_params, current_equity)
                                    
                            except Exception as e:
                                logger.error(f"Error processing trade during validation: {e}")
                                logger.error(f"Trade params: {trade_params if 'trade_params' in locals() else 'Not available'}")
                                current_position = None  # Ensure position is None if creation failed
                        
                        if current_equity <= 0:
                            logger.warning(f"Validation equity reached {current_equity:.2f}. Ending validation early.")
                            break
                        
                        # Calculate validation loss
                        image = self.agent.preprocess_screenshot(screenshot)
                        action_probs, _, _, _, _, _ = self.agent.model(image)
                        target = torch.tensor([current_position.type if current_position else 0], device=self.agent.device)
                        loss = self.agent.criterion(action_probs, target)
                        val_losses.append(loss.item())
                        
                        # Early stopping if validation loss is clearly worse
                        if len(val_losses) > 5:
                            current_avg_loss = np.mean(val_losses)
                            if current_avg_loss > self.best_val_loss * self.validation_early_stop_threshold:
                                logger.info("Stopping validation early due to high loss")
                                break
                        
                        validation_steps += 1
                
                return np.mean(val_losses) if val_losses else None, metrics.calculate_metrics()
                
            except Exception as e:
                logger.error(f"Error during validation: {e}")
                return None, None
            finally:
                self.agent.model.train()
    
    def save_best_model(self):
        """Save best model and metrics"""
        with self.profile('save_model'):
            try:
                if self.best_model_state is not None:
                    # Save model
                    model_path = os.path.join(self.model_save_dir, 'best_model.pth')
                    torch.save(self.best_model_state, model_path)
                    
                    # Save metrics
                    if self.best_metrics:
                        metrics_path = os.path.join(self.log_dir, 'validation_metrics.txt')
                        with open(metrics_path, 'w') as f:
                            f.write("Validation Metrics:\n")
                            f.write("-----------------\n")
                            f.write(f"Sharpe Ratio: {self.best_metrics['sharpe_ratio']:.4f}\n")
                            f.write(f"Sortino Ratio: {self.best_metrics['sortino_ratio']:.4f}\n")
                            f.write(f"Number of Trades: {self.best_metrics['num_trades']}\n")
                            f.write(f"Average Trade Duration: {self.best_metrics['avg_trade_duration']:.2f} steps\n")
                            f.write(f"Win Rate: {self.best_metrics['win_rate']*100:.2f}%\n")
                            f.write(f"Average Return per Trade: {self.best_metrics['avg_return']*100:.4f}%\n")
                            f.write(f"Total Return: {self.best_metrics['total_return']*100:.4f}%\n")
                            f.write(f"Stop-Loss Hits: {self.best_metrics['stop_loss_hits']}\n")
                            f.write(f"Take-Profit Hits: {self.best_metrics['take_profit_hits']}\n")
                            f.write(f"Manual Exits: {self.best_metrics['manual_exits']}\n")
                            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n")
                            f.write(f"Best Episode: {self.best_episode + 1}\n")
                            f.write("\nScreenshot Statistics:\n")
                            f.write("---------------------\n")
                            f.write(f"Total Screenshots Seen: {self.total_screenshots_seen:,}\n")
                            f.write(f"Training Screenshots: {self.training_screenshots:,}\n")
                            f.write(f"Validation Screenshots: {self.validation_screenshots:,}\n")
                            f.write("\nData Sources:\n")
                            f.write("-------------\n")
                            f.write(f"Training Data: {self.train_data_path}\n")
                            f.write(f"Validation Data: {self.val_data_path}\n")
                            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    logger.info(f"Saved best model and metrics with validation loss: {self.best_val_loss:.4f}")
                    
            except Exception as e:
                logger.error(f"Error saving best model and metrics: {e}")
    
    def plot_training_progress(self):
        """Plot training and validation losses, and equity curves"""
        with self.profile('plot_progress'):
            try:
                # Only plot if we have data
                if not self.episode_train_losses:
                    logger.info("No training data available to plot yet")
                    return
                
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
                
                episodes = range(1, len(self.episode_train_losses) + 1)
                ax1.plot(episodes, self.episode_train_losses, 'b-', label='Training Loss')
                ax1.set_title('Training Loss over Episodes')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                ax1.legend()
                
                ax2.plot(episodes, self.episode_train_losses, 'b-', label='Training Loss')
                if self.episode_val_losses:  # Only plot validation losses if we have them
                    ax2.plot(episodes, self.episode_val_losses, 'r-', label='Validation Loss')
                ax2.set_title('Training vs Validation Loss')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Loss')
                ax2.grid(True)
                ax2.legend()
                
                if self.best_equity_curve:  # Only plot equity curve if we have it
                    steps = range(len(self.best_equity_curve))
                    ax3.plot(steps, self.best_equity_curve, 'g-', label=f'Best Model (Episode {self.best_episode + 1})')
                    ax3.set_title('Equity Curve of Best Model')
                    ax3.set_xlabel('Step')
                    ax3.set_ylabel('Equity')
                    ax3.grid(True)
                    ax3.legend()
                
                plt.tight_layout()
                plot_path = os.path.join(self.log_dir, 'training_progress.png')
                plt.savefig(plot_path)
                plt.close()
                
                logger.info(f"Training progress plots saved to {plot_path}")
                
            except Exception as e:
                logger.error(f"Error plotting training progress: {e}")
    
    def __del__(self):
        """Cleanup when the environment is destroyed"""
        if self.use_profiler and self.profiler:
            self.profiler.stop_cpu_monitoring()
