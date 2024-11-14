import tkinter as tk
from tkinter import ttk
from mouse_control import MouseController
from screenshot_routine import run_screenshot_routine
import threading

class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot Control Panel")
        self.mouse_controller = MouseController()
        self.stop_flag = threading.Event()
        self.routine_running = False
        self.last_duration = None
        self.last_interval = None
        
        # Create main frame with padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Mouse Control Section
        ttk.Label(main_frame, text="Mouse Control", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        self.capture_btn = ttk.Button(main_frame, text="Capture Click Position", 
                  command=self.start_capture_click)
        self.capture_btn.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Screenshot Routine Section
        ttk.Label(main_frame, text="Screenshot Routine", font=('Arial', 12, 'bold')).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Duration input
        ttk.Label(main_frame, text="Duration (minutes):").grid(row=3, column=0, padx=5)
        self.duration_var = tk.StringVar(value="5")
        self.duration_entry = ttk.Entry(main_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.grid(row=3, column=1, padx=5)
        
        # Interval input
        ttk.Label(main_frame, text="Interval (seconds):").grid(row=4, column=0, padx=5)
        self.interval_var = tk.StringVar(value="2")
        self.interval_entry = ttk.Entry(main_frame, textvariable=self.interval_var, width=10)
        self.interval_entry.grid(row=4, column=1, padx=5)
        
        # Button frame for Start, Stop, and Restart buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Start button
        self.start_btn = ttk.Button(button_frame, text="Start Screenshot Routine", 
                                  command=self.start_screenshot_routine)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        # Stop button
        self.stop_btn = ttk.Button(button_frame, text="Stop Routine", 
                                 command=self.stop_screenshot_routine,
                                 state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)

        # Restart button
        self.restart_btn = ttk.Button(button_frame, text="Restart Routine", 
                                   command=self.restart_screenshot_routine,
                                   state='disabled')
        self.restart_btn.grid(row=0, column=2, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, pady=5)
        
    def start_capture_click(self):
        """Start the click capture process in a separate thread"""
        self.status_var.set("Waiting for click...")
        thread = threading.Thread(target=self.capture_click_thread)
        thread.daemon = True
        thread.start()
    
    def capture_click_thread(self):
        """Thread function for click capture"""
        x, y = self.mouse_controller.capture_click_position()
        self.status_var.set(f"Captured position: ({x}, {y})")
    
    def start_screenshot_routine(self):
        """Start the screenshot routine in a separate thread"""
        try:
            duration = int(self.duration_var.get())
            interval = int(self.interval_var.get())
            
            # Store the last used values
            self.last_duration = duration
            self.last_interval = interval
            
            # Reset stop flag and set running state
            self.stop_flag.clear()
            self.routine_running = True
            
            # Update UI state
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.restart_btn.config(state='disabled')
            self.capture_btn.config(state='disabled')
            self.duration_entry.config(state='disabled')
            self.interval_entry.config(state='disabled')
            
            self.status_var.set("Starting screenshot routine...")
            thread = threading.Thread(target=self.screenshot_thread, args=(duration, interval))
            thread.daemon = True
            thread.start()
            
        except ValueError:
            self.status_var.set("Please enter valid numbers")
    
    def stop_screenshot_routine(self):
        """Stop the screenshot routine"""
        self.stop_flag.set()
        self.status_var.set("Stopping routine...")
        
    def restart_screenshot_routine(self):
        """Restart the screenshot routine with the last used parameters"""
        if self.last_duration is not None and self.last_interval is not None:
            # Set the last used values
            self.duration_var.set(str(self.last_duration))
            self.interval_var.set(str(self.last_interval))
            # Start the routine
            self.start_screenshot_routine()
    
    def screenshot_thread(self, duration, interval):
        """Thread function for screenshot routine"""
        try:
            run_screenshot_routine(duration, interval, self.mouse_controller, self.stop_flag)
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            # Reset UI state
            self.routine_running = False
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.restart_btn.config(state='normal')  # Enable restart button
            self.capture_btn.config(state='normal')
            self.duration_entry.config(state='normal')
            self.interval_entry.config(state='normal')
            self.status_var.set("Screenshot routine completed")

def main():
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
