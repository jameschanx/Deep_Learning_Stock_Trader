'''
Trader Utilities
James Chan 2018
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def get_prices(ticker, start_date, end_date):
    date_range = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=date_range)
    prices = pd.read_csv(ticker + '.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'])
    prices = df.join(prices).dropna()
    prices.columns=[ticker]
    return prices

def _get_n_day_return(prices, n):
    df_n_day_return = prices.shift(-n)/(prices) - 1.0
    return df_n_day_return

def _get_x(prices, rolling_window):
    ti = TechnicalIndicators(prices, rolling_window)
    psma_vals = ti.psma()[0]
    bb_vals = ti.bb(1.5)[0]
    std_vals = ti.std()
    df_x = pd.concat([psma_vals, bb_vals, std_vals], axis=1)
    return df_x

def _get_y(prices, n):
    df_y = _get_n_day_return(prices, n)
    df_y[df_y > 0] = 1
    df_y[df_y <= 0] = 0
    return df_y

def get_xy(ticker, start_date, end_date, rolling_window, n):
    prices = get_prices(ticker, start_date, end_date)
    df_x = _get_x(prices, rolling_window)
    df_y = _get_y(prices, n)
    
    #dropna in both x and y to make sure they have the same shape.
    df_xy = pd.concat([df_x, df_y], axis = 1).dropna()
    
    x = df_xy.iloc[:, :df_xy.shape[1]-1].values
    y = df_xy.iloc[:,-1].values
    return x, y

def predictions_to_trades(ticker, predictions, start_date, end_date, n, actual_start):
    #get prices so we can have a clean trade dataframe
    prices = get_prices(ticker, start_date, end_date)
    trades = prices.copy()
    trades.values[:,:] = 0
    
    count_down = 1
    holding = 0
    for i, prediction in enumerate(predictions):
        i = i + actual_start
        if count_down > 0:
            count_down -= 1
        if count_down == 0:
            if prediction == 1:
                if(holding < 0):
                    trades.values[i,:] = 2
                    holding += 2
                    count_down = n
                elif(holding > 0):
                    pass
                else:
                    trades.values[i,:] = 1
                    holding += 1
                    count_down = n            
            else:
                if(holding < 0):
                    pass
                elif(holding > 0):
                    trades.values[i,:] = -2
                    holding -= 2
                    count_down = n
                else:
                    trades.values[i,:] = -1
                    holding -= 1
                    count_down = n
    return trades


class TechnicalIndicators:
    def __init__(self, prices, rolling_window=21):
        self.rolling_window = rolling_window
        self.prices = prices
        self.ticker = prices.columns.values[0]
        pass
    
    def psma(self):
        sma = pd.DataFrame.rolling(self.prices, self.rolling_window).mean()
        psma = self.prices/sma.values
        return psma, sma
    
    def std(self):
        std = pd.DataFrame.rolling(self.prices, self.rolling_window).std()
        return std
        
    def bb(self, sigma=1.5):
        sma = pd.DataFrame.rolling(self.prices, self.rolling_window).mean()
        std = pd.DataFrame.rolling(self.prices, self.rolling_window).std()
        up_band = sma + sigma * std.values
        low_band = sma - sigma * std.values
        return ((self.prices - low_band) / (up_band - low_band)), up_band, low_band
    
    def example_plots(self, plot_size = (8,6), output = False):
        #plt.switch_backend('agg')
        prices = self.prices
    
        #------------------------------PSMA PLOT----------------------------------
        psma, sma = self.psma()
        plt.figure("SMA", plot_size)
        gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
        ax0 = plt.subplot(gs[0])
        plt.title("Price to SMA Ratio for {} ({}-Day Window))".format(self.ticker, self.rolling_window))
        ax1 = plt.subplot(gs[1], sharex=ax0)
        plt.xlabel("Date")
        ax0.plot(prices, label="Price", color='gray')
        ax0.plot(sma, label="SMA",color='orange')
        ax0.set_ylabel("Price")
        ax0.grid()
        ax0.legend()
        ax1.plot(psma, label="Price to SMA Ratio")
        ax1.set_ylabel("P/SMA Ratio")
        ax1.set_ylim((.8, 1.2))
        ax1.fill_between([psma.index[0],psma.index[-1]], [0,0], [.95, .95], alpha = '.50', color='limegreen')
        ax1.fill_between([psma.index[0],psma.index[-1]], [1.05,1.05,], [2, 2], alpha = '.50', color='lightcoral')
        sell_signal = (psma.where(psma > 1.05).dropna())
        buy_signal = (psma.where(psma < .95).dropna())
        ax0.scatter(buy_signal.index, prices.loc[buy_signal.index], marker='^', color='limegreen', s=20)
        ax0.scatter(sell_signal.index, prices.loc[sell_signal.index], marker='v', color='indianred', s=20)
        ax1.grid()
        ax1.legend()
        plt.show()
        if output: 
            plt.savefig('PSMA.png')
        
        #-------------------------------STD PLOT----------------------------------
        std = self.std()
        std_over = std.where(std > 20)
        plt.figure("STD", plot_size)
        plt.xlim(prices.index[0], prices.index[-1])
        plt.title("Volatiliy for {} ({}-Day Window)".format(self.ticker, self.rolling_window))
        plt.xlabel("Date")
        plt.ylabel("Rolling Standard Deviation")
        plt.axhline(y=20)
        plt.plot(std, color='gray')
        plt.fill_between(std_over.index, [20 for _ in range(std_over.shape[0])], std_over.values.T[0], alpha='.5')
        plt.setp(plt.gca().xaxis.get_majorticklabels(),'rotation', 30)
        plt.grid()
        plt.show()
        if output:
            plt.savefig('STD.png')
        
        #-------------------------------BB PLOT----------------------------------
        bb, ub, lb = self.bb(1.5)
        ub_signal = ub / prices
        lb_signal = prices / lb
        sell_signal = (ub_signal.where(ub_signal < 1.0).dropna())
        buy_signal = (ub_signal.where(lb_signal < 1.0).dropna())
        plt.figure("BB", plot_size)
        y_up_lim = ub.iloc[:,[0]].dropna().max().values[0] * 1.10
        y_low_lim = lb.iloc[:,[0]].dropna().min().values[0] * .90
        plt.ylim((y_low_lim, y_up_lim))
        plt.title("Bollinger Bands for {} ({}-Day Window)".format(self.ticker, self.rolling_window))
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.fill_between(prices.index, ub.values.T[0], lb.values.T[0], alpha = '.20', color='gray')
        plt.plot(ub, label="Upper Band / Sell Opportunity", color='lightcoral')
        plt.plot(lb, label="Lower Band / Buy Opportunity", color='limegreen')
        plt.grid()
        plt.fill_between(prices.index, (ub/prices.iloc[0,:]).values.T[0], (lb/prices.iloc[0,:]).values.T[0], alpha = '.20', color='gray')
        for i in buy_signal.index:
            plt.axvline(i, 0, (prices.loc[i] - y_low_lim)/(y_up_lim - y_low_lim), color='lightgreen')
        for i in sell_signal.index:
            plt.axvline(i, (prices.loc[i] - y_low_lim)/(y_up_lim - y_low_lim), 1, color='lightpink')
        plt.plot(prices, label="Price",color='slategray')
        plt.legend()
        plt.show()
        if output:
            plt.savefig('BB.png')
        
class BackTester:
    def __init__(self, starting_cash = 1000000, holding_limit = 1000):
        self.starting_cash  = starting_cash 
        self.holding_limit = holding_limit
        
    def simulate_portfolio(self, df_trades):
        df_trades = df_trades * self.holding_limit
        start_date, end_date = df_trades.index[[0, -1]]
        date_range = pd.date_range(start_date, end_date)
        ticker = df_trades.columns.values[0]
        df = pd.DataFrame(index=date_range)
        prices = pd.read_csv(ticker + '.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'])
        prices = df.join(prices).dropna()
        prices.columns=[ticker]
        df_trades['CASH'] = 0.0
        for date, row in df_trades.iterrows():
            shares = row[ticker]    
            cash = - shares * prices.loc[date,[ticker]].values[0]
            df_trades.loc[date,['CASH']] += cash
        df_trades.loc[start_date,['CASH']] += self.starting_cash
        holdings = np.cumsum(df_trades, axis=0)
        prices['CASH'] = 1.0
        portvals = holdings * prices
        portvals = np.sum(portvals, axis=1)
        return portvals
    
    def backtest(self, df_trades, plot_title = 'Untitled', algorithm_title = 'Untitled Algorithm', benchmark = True, plot_size=(8,6), output=False):
        start_date, end_date = df_trades.index[[0, -1]]
        ticker = df_trades.columns.values[0]

        plt.figure("in-sample", plot_size)
        gs = gridspec.GridSpec(2,1, height_ratios=[4,1])
        
        #main plot for portfolio values
        ax0 = plt.subplot(gs[0])
        plt.title(plot_title)
        plt.ylabel("Portfolio Value")
        if benchmark:
            df_trades_bench = get_prices(ticker, start_date, end_date)
            df_trades_bench[:] = 0.
            df_trades_bench.iloc[0] = 1
            port_vals_bench = self.simulate_portfolio(df_trades_bench)
            ax0.plot(port_vals_bench/port_vals_bench.values[0], color="gray", label="Benchmark")            
        port_vals = self.simulate_portfolio(df_trades)

        ax0.plot(port_vals/port_vals.values[0], color="cornflowerblue", label=algorithm_title)
        ax0.set_ylabel("Normalized Portfolio Value")
        ax0.grid()
        ax0.legend()
        
        #subplot for order entries
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Date")
        buy_line = df_trades[ticker][df_trades[ticker] > 0].index
        sell_line = df_trades[ticker][df_trades[ticker] < 0].index
        ax1.set_ylabel("Order Entry")
#        ax1.set_ylim((0, 0))
        for i, line in enumerate(buy_line):
            if i == 0:
                ax1.axvline(line, color='limegreen', label='Buy')
            else:
                ax1.axvline(line, color='limegreen')
        for i, line in enumerate(sell_line):
            if i == 0:
                ax1.axvline(line, color='indianred', label='Sell')
            else:
                ax1.axvline(line, color='indianred')
        ax1.legend()
        ax1.get_yaxis().set_ticks([])
        plt.show()
        if output:
            plt.savefig(plot_title + '.png')

if __name__=='__main__':
    print('ni hao')
#    start_date = pd.datetime(2011,1,1)
#    end_date = pd.datetime(2012,1,1)
#    ticker = 'GOOG'
#    prices = get_prices(ticker, start_date, end_date)
#    ti = TechnicalIndicators(prices)
#    ti.example_plots(plot_size = (8,6))