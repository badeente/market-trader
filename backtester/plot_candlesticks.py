import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import io

def plot_candlesticks(df, num_entries=50):
    """
    Create a candlestick plot with multiple EMAs from DataFrame data
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing OHLCV data with timestamp index
    num_entries (int): Number of last entries to plot (default: 50)
    
    Returns:
    bytes: PNG image data stored in memory
    """
    try:
        # Ensure timestamp is datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index, format='mixed')

        # Get last n entries
        df_last_n = df.tail(num_entries)

        # Calculate EMAs
        df_last_n['EMA20'] = df_last_n['close'].ewm(span=20, adjust=False).mean()
        df_last_n['EMA50'] = df_last_n['close'].ewm(span=50, adjust=False).mean()
        df_last_n['EMA100'] = df_last_n['close'].ewm(span=100, adjust=False).mean()
        df_last_n['EMA200'] = df_last_n['close'].ewm(span=200, adjust=False).mean()

        # Define style
        style = mpf.make_mpf_style(
            base_mpf_style='charles',
            figcolor='black',
            facecolor='black',
            edgecolor='white',
            gridcolor='gray',
            gridstyle=':',
            rc={'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'figure.facecolor': 'black'},
            marketcolors=mpf.make_marketcolors(
                up='green',
                down='red',
                edge='inherit',
                wick='inherit',
                volume='in',
                ohlc='inherit'
            )
        )

        # Create the additional plots for EMAs
        ema_plots = [
            mpf.make_addplot(df_last_n['EMA20'], color='red', width=1),
            mpf.make_addplot(df_last_n['EMA50'], color='orange', width=1),
            mpf.make_addplot(df_last_n['EMA100'], color='green', width=1),
            mpf.make_addplot(df_last_n['EMA200'], color='yellow', width=1)
        ]

        # Create in-memory buffer
        buf = io.BytesIO()
        
        # Create the candlestick plot with EMAs
        fig, axes = mpf.plot(df_last_n, 
                           type='candle',
                           style=style,
                           title=f'BTC-USDT Last {num_entries} Candlesticks (5min) with EMAs',
                           ylabel='Price (USDT)',
                           addplot=ema_plots,
                           returnfig=True)
        
        # Save to buffer
        fig.savefig(buf, format='png')
        plt.close(fig)  # Clean up the figure
        
        # Get the bytes
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please ensure the DataFrame contains valid timestamp and OHLCV data")
        return None

# Example usage if run directly
if __name__ == "__main__":
    # Example with CSV data
    df = pd.read_csv('data/btc/5min/dev.csv')
    png_bytes = plot_candlesticks(df)
    if png_bytes:
        # Save to file just for testing
        with open('candlestick_plot.png', 'wb') as f:
            f.write(png_bytes)
