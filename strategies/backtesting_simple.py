from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from strategies import two_legged_pullback_strategy

from backtesting.test import SMA, GOOG


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)
        self.strategie = two_legged_pullback_strategy.TwoLeggedPullbackStrategy()

    def next(self):
        prediction = self.strategie.predict()
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot()