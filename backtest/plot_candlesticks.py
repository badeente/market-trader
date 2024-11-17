import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

try:
    # Read the CSV file
    df = pd.read_csv('C:\\workspace\\market-trader\\backtest\\data\\btc\\5min\BTC-USDT.csv')

    # Convert timestamp to datetime - handles both standard timestamps and .000 format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df.set_index('timestamp', inplace=True)

    # Get last 50 entries
    df_last_50 = df.tail(100)

    # Calculate EMAs
    df_last_50['EMA20'] = df_last_50['close'].ewm(span=20, adjust=False).mean()
    df_last_50['EMA50'] = df_last_50['close'].ewm(span=50, adjust=False).mean()
    df_last_50['EMA100'] = df_last_50['close'].ewm(span=100, adjust=False).mean()
    df_last_50['EMA200'] = df_last_50['close'].ewm(span=200, adjust=False).mean()

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
        mpf.make_addplot(df_last_50['EMA20'], color='red', width=1),
        mpf.make_addplot(df_last_50['EMA50'], color='orange', width=1),
        mpf.make_addplot(df_last_50['EMA100'], color='green', width=1),
        mpf.make_addplot(df_last_50['EMA200'], color='yellow', width=1)
    ]

    # Create the candlestick plot with EMAs
    mpf.plot(df_last_50, 
             type='candle',
             style=style,
             title='BTC-USDT Last 50 Candlesticks (5min) with EMAs',
             ylabel='Price (USDT)',
             addplot=ema_plots,
             savefig='candlestick_plot.png')
    
    print("Plot successfully created as 'candlestick_plot.png'")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Please ensure the CSV file contains valid timestamp and OHLCV data")
