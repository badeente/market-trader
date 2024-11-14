import keyboard
import time
import sys

def send_ctrl_shift_s():
    """
    Sends the keyboard combination Ctrl + Shift + S
    This simulates pressing all three keys simultaneously
    """
    keyboard.send("ctrl+shift+s")

def send_ctrl_alt_s():
    """
    Sends the keyboard combination Alt + S
    This simulates pressing both keys simultaneously
    Then performs a mouse click
    """
    keyboard.send('ctrl+alt+s')

def send_alt_r():
    """
    Sends the keyboard combination Alt + R
    This simulates pressing both keys simultaneously
    """
    keyboard.send('alt+r')

def send_arrow_left():
    """
    Sends a left arrow key stroke
    This simulates pressing the left arrow key
    """
    keyboard.send('left')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python keyboard_control.py <seconds>")
        print("Example: python keyboard_control.py 5")
        sys.exit(1)
    
    try:
        seconds = int(sys.argv[1])
        print(f"Waiting {seconds} seconds before sending Ctrl+Shift+S...")
        time.sleep(seconds)
        
        send_ctrl_alt_s()
        print("Keyboard combination sent!")
    except ValueError:
        print("Error: Please provide a valid number of seconds")
        sys.exit(1)
