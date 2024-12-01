import os
import time
import logging
import sys
sys.path.append('.')  # Add current directory to path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "backtester", "data", "btc", "5min", "BTCUSDT_train.csv")
VAL_DATA_PATH = os.path.join(BASE_DIR, "backtester", "data", "btc", "5min", "BTCUSDT_validation.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

def main():
    """Main training loop with improved logging and error handling"""
    try:
        from training_environment import TrainingEnvironment
        
        start_time = time.time()
        logger.info("Starting training process")
        
        # Initialize training environment with default paths
        env = TrainingEnvironment(
            train_data_path=TRAIN_DATA_PATH,
            val_data_path=VAL_DATA_PATH,
            model_save_dir=MODEL_SAVE_DIR
        )
        
        # Run just one episode to verify screenshot counting
        num_episodes = 2
        for episode in range(num_episodes):
            logger.info(f"Starting episode {episode + 1}/{num_episodes}")
            
            avg_reward, train_loss, val_loss, final_equity = env.train_episode(episode, num_steps=400)
            
            logger.info(
                f"Episode {episode + 1} completed:\n"
                f"Average Reward: {avg_reward:.4f}\n"
                f"Training Loss: {train_loss:.4f}\n"
                f"Validation Loss: {val_loss:.4f}\n"
                f"Final Equity: {final_equity:.2f}"
            )
        
        env.plot_training_progress()
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        logger.info(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Display the validation metrics file content
        metrics_path = os.path.join(MODEL_SAVE_DIR, 'logs', 'validation_metrics.txt')
        if os.path.exists(metrics_path):
            logger.info("\nValidation Metrics:")
            with open(metrics_path, 'r') as f:
                print(f.read())
            
    except Exception as e:
        logger.error(f"Error in main training loop: {e}")
        raise

if __name__ == "__main__":
    main()
