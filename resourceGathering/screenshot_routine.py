import time
import keyboard
from keyboard_control import send_ctrl_alt_s
from clipboard_screenshot import save_clipboard_screenshot
from keyboard_control import send_arrow_left
from datetime import datetime, timedelta

def run_screenshot_routine(interval_seconds, stop_flag=None, ):
    """
    Run the screenshot routine for a specified duration with custom intervals.
    
    Args:
        duration_minutes (int): How long the routine should run in minutes
        interval_seconds (int): Time interval between screenshots in seconds
        stop_flag (threading.Event, optional): Event to signal stopping the routine
    """
    
    print("Waiting 5 seconds before starting...")
    print("Press 'q' at any time to stop the routine")
    time.sleep(5)  # Wait 10 seconds before starting
    
    # Calculate end time based on duration parameter
    
    
    running = True
    count = 0
    while running:
        # Check for stop conditions
        if keyboard.is_pressed('q') or (stop_flag and stop_flag.is_set()):
            print("\nStopping routine...")
            running = False
            break
            
        
        #Send keyboard shortcut
        send_ctrl_alt_s()

        # Wait 1 second
        time.sleep(1)
        send_arrow_left()
              
        # Wait for the specified interval
        time.sleep(interval_seconds)
    
    print("Screenshot routine completed")

if __name__ == "__main__":
    run_screenshot_routine(interval_seconds=3)
