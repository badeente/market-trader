import datetime
import pandas_ta as ta
import pandas as pd
from enum import Enum


class Backtester():
    equity = 10000
    df = None
    currentPosition = 0
    # Do as much initial computation as possible
    def init(self, df):
        self.df = df


    # Step through bars one by one
    # Note that multiple buys are a thing here

    def next(self, command):
        # No changes just return equity
        if command == self.currentPosition:     
            return self.equity  
        
        if command == 0 :

            self.currentPosition = 0           
        elif command == 1:
            print("buy")
        elif command == 2:
            print("sell")
        return self.equity