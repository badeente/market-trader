import mouse
import time

class MouseController:
    def __init__(self):
        self.click_x = 0
        self.click_y = 0

    def click_at_position(self, x=None, y=None):
        """
        Performs a mouse click at specified coordinates.
        If no coordinates provided, uses stored coordinates.
        
        Args:
            x (int): X coordinate (optional)
            y (int): Y coordinate (optional)
        """
        click_pos_x = x if x is not None else self.click_x
        click_pos_y = y if y is not None else self.click_y
        
        # Move mouse to position and click
        mouse.move(click_pos_x, click_pos_y)
        time.sleep(0.1)  # Small delay to ensure mouse has moved
        mouse.click()
        
    def capture_click_position(self):
        """
        Waits for user to perform a mouse click and saves those coordinates.
        Returns the captured coordinates.
        """
        print("Please click at the desired position...")
        
        # Record the next click event
        mouse.wait(button='left')
        self.click_x, self.click_y = mouse.get_position()
        
        print(f"Captured click position: ({self.click_x}, {self.click_y})")
        return self.click_x, self.click_y

if __name__ == "__main__":
    # Example usage
    controller = MouseController()
    
    # Capture click position
    controller.capture_click_position()
    
    # Wait a moment
    time.sleep(2)
    
    # Click at the captured position
    controller.click_at_position()
    
    # Or click at specific coordinates
    # controller.click_at_position(100, 200)
