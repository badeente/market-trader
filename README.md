# market-trader


ToDo's:
-> in backtesting_simple or backtest_example:
- create Screenshot of candlestick chart when a strategie is executing an order
- create a second screenshot of candlestick chart after X bars 
- save both screenshots in a seperate folder
- save additional info file with with data that did lead to the oder 
- save type of oder in info file
- check in backtesting_example if statistics section is providing good information

-> in tow_legged_pullback_strategy:
- play with count of bars to determine two legged pullback
- adjust order action to only execute order according to determined trend
    - uptrend => buy downtrend => sell.

=> trend_classification.py
- debug trendline value and figure out what determines a trend best or at least good

=> new file
- implement utils class to execute simple but often needed methods

=> screenshot service
- check if screenshot are easy to create and get with only providing candlestick chart data
    Done 
