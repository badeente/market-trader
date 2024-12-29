from csv_loader_candlestickchart import CsvLoaderCandlestickChart

def test_loader():
    # Create loader instance
    loader = CsvLoaderCandlestickChart()
    
    # Load example data
    df = loader.load_csv("../data/dev.csv")
    
    # Print basic information about the loaded data
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Verify datetime conversion
    print("\nDatetime dtype:", df['datetime'].dtype)
    
    return df

if __name__ == "__main__":
    test_loader()
