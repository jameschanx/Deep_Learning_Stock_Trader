'''
Neural Network Trader
James Chan 2018
'''

import pandas as pd
import matplotlib.pyplot as plt
from trader_utils import BackTester, predictions_to_trades, get_xy

class DeepLearningTrader:
    def __init__(self):
        self.n = 1
        self.rolling_window = 21
#        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20)
        self.ticker = None

    def fit(self, ticker, start_date, end_date): 
        self.ticker = ticker
        train_x, train_y = get_xy(ticker, start_date, end_date, self.rolling_window, self.n)
#        self.learner.addEvidence(train_x, train_y)
    
    def generate_trades(self, start_date, end_date):
        test_x, _ = get_xy(self.ticker, start_date, end_date, self.rolling_window, self.n)
        predictions = self.learner.query(test_x)
        actual_start = self.rolling_window - 1
        trades = predictions_to_trades(self.ticker, predictions, start_date, end_date, self.n, actual_start)
        return trades
        
if __name__=="__main__":
    #initialize
    nnt = DeepLearningTrader()
    starting_cash = 100000
    holding_limit = 1000
    btr = BackTester(starting_cash, holding_limit)
    plt.switch_backend('agg')
    
    #in-sample
    start_date = pd.datetime(2011,1,1)
    end_date = pd.datetime(2011,12,31)
    
    #train model
    ticker = 'GOOG'
    nnt.fit(ticker, start_date, end_date)
    
    #generate trade in-sample
    df_trades = nnt.generate_trades(start_date, end_date) 
    
    plot_title = 'Deep Learning Trader for {}, In-Sample'.format(ticker)
    algorithm_title = 'Deep Learning Trader'
    btr.backtest(df_trades, plot_title, algorithm_title, benchmark=True, plot_size=(8,6))
    
    #out-of-sample
    start_date = pd.datetime(2012,1,1)
    end_date = pd.datetime(2012,12,31)
    
    #generate trade out-of-sample
    df_trades = nnt.generate_trades(start_date, end_date) 
    plot_title = 'Deep Learning Trader for , Out-of-Sample'.format(ticker)
    algorithm_title = 'Deep Learning Trader'
    btr.backtest(df_trades, plot_title, algorithm_title, benchmark=True, plot_size=(8,6))    
