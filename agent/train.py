import os
from model import TradingAgent
import time
from collections import deque
import random

class TrainingEnvironment:
    def __init__(self, screenshots_dir, model_save_dir="saved_models"):
        self.agent = TradingAgent()
        self.screenshots_dir = screenshots_dir
        self.model_save_dir = model_save_dir
        self.memory = deque(maxlen=1000)  # Store recent experiences
        self.episode_rewards = []
        
        os.makedirs(model_save_dir, exist_ok=True)
    
    def get_latest_screenshots(self, n=5):
        """Get the n most recent screenshots"""
        screenshots = []
        for file in sorted(os.listdir(self.screenshots_dir))[-n:]:
            if file.endswith('.png'):
                screenshots.append(os.path.join(self.screenshots_dir, file))
        return screenshots
    
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
            # Get latest screenshots
            screenshots = self.get_latest_screenshots()
            if not screenshots:
                time.sleep(1)
                continue
            
            # Make prediction
            latest_screenshot = screenshots[-1]
            decision, minutes = self.agent.predict(latest_screenshot)
            
            # In a real environment, you would wait for actual outcome
            # For this example, we'll simulate it
            # You should replace this with actual market data
            actual_outcome = ("green" if random.random() > 0.5 else "red", 
                            random.randint(1, 10))
            
            # Calculate reward
            reward = self.calculate_reward(decision, minutes, actual_outcome)
            episode_rewards.append(reward)
            
            # Store experience
            self.memory.append((latest_screenshot, reward))
            
            # Train on a batch of experiences
            if len(self.memory) >= 32:
                batch = random.sample(self.memory, 32)
                for screenshot, reward in batch:
                    loss = self.agent.train_step(screenshot, reward)
            
            time.sleep(1)  # Wait before next prediction
        
        # Save model after episode
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        self.episode_rewards.append(avg_reward)
        
        model_path = os.path.join(self.model_save_dir, 
                                 f'model_reward_{avg_reward:.2f}.pth')
        self.agent.save_model(model_path)
        
        return avg_reward

def main():
    screenshots_dir = "../screenshots/gatherer"  # Adjust path as needed
    env = TrainingEnvironment(screenshots_dir)
    
    num_episodes = 10
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}/{num_episodes}")
        avg_reward = env.train_episode()
        print(f"Episode {episode + 1} completed with average reward: {avg_reward:.2f}")

if __name__ == "__main__":
    main()
