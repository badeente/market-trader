import os
from model import TradingAgent
import time

def main():
    # Initialize agent with a trained model if available
    model_dir = "saved_models"
    model_path = None
    
    # Find the latest model file
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
            model_path = os.path.join(model_dir, latest_model)
    
    agent = TradingAgent(model_path)
    screenshots_dir = "../screenshots/gatherer"  # Adjust path as needed
    
    print("Starting prediction loop...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Get latest screenshot
            screenshots = sorted(os.listdir(screenshots_dir))
            if screenshots:
                latest_screenshot = os.path.join(screenshots_dir, screenshots[-1])
                
                # Make prediction
                decision, minutes = agent.predict(latest_screenshot)
                
                print(f"\nPrediction for {os.path.basename(latest_screenshot)}:")
                print(f"Decision: {decision}")
                print(f"Minutes: {minutes}")
            
            time.sleep(1)  # Wait before next prediction
            
    except KeyboardInterrupt:
        print("\nStopping prediction loop")

if __name__ == "__main__":
    main()
