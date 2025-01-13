import pandas as pd
from datetime import datetime

class CsvLoaderCandlestickChart:
    """
    A class to load candlestick data from CSV files into pandas DataFrames.
    Handles different timestamp formats and ensures proper data structure.
    """
    
    def __init__(self):
        """Initialize the loader with default settings"""
        self.df = None
        self.required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load candlestick data from a CSV file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file containing candlestick data
            
        Returns:
            pd.DataFrame: DataFrame containing the candlestick data with properly formatted timestamps
            
        Raises:
            ValueError: If required columns are missing or data format is invalid
        """
        try:
            # Read CSV file
            self.df = pd.read_csv(file_path)
            
            # Verify required columns exist
            missing_cols = [col for col in self.required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert datetime column to pandas datetime
            # This will automatically handle various timestamp formats
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            
            # Ensure numeric columns are float type
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Sort by datetime
            self.df = self.df.sort_values('datetime')
            
            return self.df
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the loaded DataFrame.
        
        Returns:
            pd.DataFrame: The loaded candlestick data
            
        Raises:
            ValueError: If no data has been loaded yet
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        return self.df
    
