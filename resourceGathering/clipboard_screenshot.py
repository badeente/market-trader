from PIL import ImageGrab
from datetime import datetime
import os

def save_clipboard_screenshot(save_path):
    try:
        # Get image from clipboard
        image = ImageGrab.grabclipboard()
        
        if image is None:
            print("No image found in clipboard!")
            return False
        
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        
        # Combine path and filename
        full_path = os.path.join(save_path, filename)
        
        # Save the image
        image.save(full_path, 'PNG')
        print(f"Screenshot saved as: {os.path.abspath(full_path)}")
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python clipboard_screenshot.py <save_path>")
        print("Example: python clipboard_screenshot.py C:\\Screenshots")
        sys.exit(1)
        
    save_path = sys.argv[1]
    save_clipboard_screenshot(save_path)
