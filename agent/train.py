import os
from model import TradingAgent
import time
from collections import deque
import random
import sys
sys.path.append('..')  # Add parent directory to path
from backtest.data_provider import DataProvider
from backtest.plot_candlesticks import plot_candlesticks

class TrainingEnvironment:
    def __init__(self, data_path="backtest/data/btc/5min/dev.csv", model_save_dir="saved_models"):
        self.agent = TradingAgent()
        self.data_provider = DataProvider(data_path)
        self.model_save_dir = model_save_dir
        self.memory = deque(maxlen=1000)  # Store recent experiences
        self.episode_rewards = []
        
        os.makedirs(model_save_dir, exist_ok=True)
    
    def get_latest_screenshots(self):
        """Get the latest market data as a PNG screenshot"""
        # Get next window of 200 entries from data provider
        window = self.data_provider.get_next_window()
        if window is None:
            return None
            
        # Generate PNG screenshot from the data
        png_data = plot_candlesticks(window, num_entries=200)
        return png_data
    
    def calculate_reward(self, decision, minutes, actual_outcome):
        """
        Calculate reward based on the accuracy of prediction
        actual_outcome should be a tuple of (actual_direction, actual_minutes)
        """
        direction_reward = 1 if decision == actual_outcome[0] else -1
        
        # Calculate minutes reward based on how close the prediction was
        minutes_diff = abs(minutes - actual_outcome[1])
        minutes_reward = max(0, 1 - minutes_diff/10)  # Scale down penalty for minute differences
        
        total_reward = direction_reward + minutes_reward
        return total_reward
    
    def train_episode(self, num_steps=100):
        """Train for one episode"""
        episode_rewards = []
        
        for _ in range(num_steps):
            # Get latest screenshot
            screenshot_data = self.get_latest_screenshots()
            if screenshot_data is None:
                print("No more data available")
                break
            
            # Make prediction
            decision, minutes = self.agent.predict(screenshot_data)
            
            # In a real environment, you would wait for actual outcome
            # For this example, we'll simulate it
            # You should replace this with actual market data
            actual_outcome = ("green" if random.random() > 0.5 else "red", 
                            random.randint(1, 10))
            
            # Calculate reward
            reward = self.calculate_reward(decision, minutes, actual_outcome)
            episode_rewards.append(reward)
            
            # Store experience
            self.memory.append((screenshot_data, reward))
            
            # Train on a batch of experiences
            if len(self.memory) >= 32:
                batch = random.sample(self.memory, 32)
                for screenshot, reward in batch:
                    loss = self.agent.train_step(screenshot, reward)
            
            time.sleep(1)  # Wait before next prediction
        
        # Save model after episode
        if episode_rewards:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            self.episode_rewards.append(avg_reward)
            
            model_path = os.path.join(self.model_save_dir, 
                                     f'model_reward_{avg_reward:.2f}.pth')
            self.agent.save_model(model_path)
            
            return avg_reward
        return 0.0

def main():
    env = TrainingEnvironment()
    
    num_episodes = 10
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}/{num_episodes}")
        avg_reward = env.train_episode()
        print(f"Episode {episode + 1} completed with average reward: {avg_reward:.2f}")

if __name__ == "__main__":
    main()
