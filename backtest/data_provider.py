import pandas as pd

class DataProvider:
    def __init__(self, csv_path):
        """
        Initialize DataProvider with CSV data
        
        Parameters:
        csv_path (str): Path to the CSV file containing trading data
        """
        self.df = pd.read_csv(csv_path)
        self.counter = 0
        self.window_size = 200
        
    def get_next_window(self):
        """
        Returns the next window of 200 entries based on counter
        
        Returns:
        pandas.DataFrame: DataFrame containing 200 entries or None if no more data
        """
        start_idx = self.counter
        end_idx = start_idx + self.window_size
        
        # Check if we have enough data left
        if end_idx > len(self.df):
            return None
            
        # Get the window
        window = self.df.iloc[start_idx:end_idx].copy()
        
        # Increment counter for next call
        self.counter += 1
        
        return window

if __name__ == "__main__":
    # Example usage
    provider = DataProvider("data/btc/5min/dev.csv")
    
    # Get windows until None is returned
    window_count = 0
    while True:
        window = provider.get_next_window()
        if window is None:
            print("No more data available")
            break
            
        print(f"Window {window_count}:")
        print(f"Start index: {provider.counter-1}")
        print(f"End index: {provider.counter-1 + provider.window_size}")
        print(f"Shape: {window.shape}")
        print("-" * 50)
        window_count += 1
