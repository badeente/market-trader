import pandas as pd
import numpy as np
from pathlib import Path
import os

def split_by_random_weeks(file_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    try:
        print(f"Reading file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"Initial data shape: {df.shape}")
        
        if len(df) == 0:
            raise ValueError("Input file is empty")
        
        # Convert timestamp to datetime using flexible parsing
        print("Converting timestamps...")
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        except ValueError:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
            except ValueError:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        
        # Add week identifier
        print("Adding week identifiers...")
        df['year_week'] = df['timestamp'].dt.strftime('%Y-%V')  # ISO week number format
        
        # Get unique weeks and shuffle them
        unique_weeks = df['year_week'].unique()
        print(f"Found {len(unique_weeks)} unique weeks")
        
        if len(unique_weeks) < 3:
            raise ValueError(f"Not enough weeks ({len(unique_weeks)}) for splitting. Need at least 3 weeks.")
        
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_weeks)
        
        # Calculate number of weeks for each split
        n_weeks = len(unique_weeks)
        n_train_weeks = max(1, int(n_weeks * train_ratio))
        n_val_weeks = max(1, int(n_weeks * val_ratio))
        n_test_weeks = max(1, n_weeks - n_train_weeks - n_val_weeks)
        
        print(f"Splitting into {n_train_weeks}/{n_val_weeks}/{n_test_weeks} weeks")
        
        # Split weeks into train, validation, and test sets
        train_weeks = unique_weeks[:n_train_weeks]
        val_weeks = unique_weeks[n_train_weeks:n_train_weeks + n_val_weeks]
        test_weeks = unique_weeks[n_train_weeks + n_val_weeks:]
        
        # Create dataframes for each split
        print("Creating split dataframes...")
        train_df = df[df['year_week'].isin(train_weeks)].copy()
        val_df = df[df['year_week'].isin(val_weeks)].copy()
        test_df = df[df['year_week'].isin(test_weeks)].copy()
        
        # Sort each split chronologically
        train_df = train_df.sort_values('timestamp')
        val_df = val_df.sort_values('timestamp')
        test_df = test_df.sort_values('timestamp')
        
        # Remove helper column
        train_df = train_df.drop('year_week', axis=1)
        val_df = val_df.drop('year_week', axis=1)
        test_df = test_df.drop('year_week', axis=1)
        
        # Generate output filenames
        file_path = Path(file_path)
        base_name = file_path.stem
        output_dir = file_path.parent
        
        print(f"Output directory: {output_dir}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits with absolute paths
        train_path = output_dir / f"{base_name}_train.csv"
        val_path = output_dir / f"{base_name}_validation.csv"
        test_path = output_dir / f"{base_name}_test.csv"
        
        print(f"\nSaving files:")
        print(f"Train path: {train_path}")
        train_df.to_csv(train_path, index=False)
        print(f"Validation path: {val_path}")
        val_df.to_csv(val_path, index=False)
        print(f"Test path: {test_path}")
        test_df.to_csv(test_path, index=False)
        
        # Verify files were created
        for path in [train_path, val_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Failed to create output file: {path}")
            if os.path.getsize(path) == 0:
                raise ValueError(f"Created file is empty: {path}")
        
        # Print detailed information
        print("\nDetailed Split Information:")
        print(f"Total weeks: {len(unique_weeks)}")
        print(f"Training weeks: {len(train_weeks)} ({len(train_weeks)/len(unique_weeks)*100:.1f}%)")
        print(f"Validation weeks: {len(val_weeks)} ({len(val_weeks)/len(unique_weeks)*100:.1f}%)")
        print(f"Test weeks: {len(test_weeks)} ({len(test_weeks)/len(unique_weeks)*100:.1f}%)")
        
        print("\nRows per split:")
        print(f"Total rows: {len(df)}")
        print(f"Training rows: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation rows: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test rows: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        print("\nTime ranges:")
        print(f"Training: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"Validation: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        print(f"Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        # Print selected weeks for verification
        print("\nSelected weeks:")
        print("Training weeks:", sorted(train_weeks))
        print("Validation weeks:", sorted(val_weeks))
        print("Test weeks:", sorted(test_weeks))
        
        return (len(train_df), len(val_df), len(test_df))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    file_path = os.path.join("backtester", "data", "btc", "5min", "BTCUSDT.csv")
    print(f"Working directory: {os.getcwd()}")
    print(f"Full input path: {os.path.abspath(file_path)}")
    split_by_random_weeks(file_path)
