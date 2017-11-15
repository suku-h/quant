import glob
import pandas as pd


path = r'F:\data\nse500\indices'
allFiles = glob.glob(path + "\*.csv")


def getStocks(file='F:/data/nse500/indices/ind_nifty500list.csv'):
    df = pd.read_csv(file)
    stocks = df['Symbol'].values
    for i in range(len(stocks)):
        if stocks[i] == 'FAGBEARING':
            stocks[i] = 'SCHAEFFLER'

    return stocks


def getData():
    data = {}
    stocks = getStocks()
    for count, ticker in enumerate(stocks):
        total_df = pd.read_csv("F:/data/nse500/selected/{}.csv".format(ticker))
        total_df.reset_index(inplace=True)
        df = total_df[['Symbol', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'VWAP']]
        df.set_index(['Symbol', 'Date'], inplace=True)

        # make Volume as double as talib needs it as double/float
        df['Volume'] = df['Volume'].astype(float)
        data[ticker] = df

    return data