import bs4 as bs
import requests
import pickle
import datetime as dt
import os
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from Collections import Counter

pickle_name = "sp500tickers.pickle"


def save_SP500_tickers():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text)
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        # ticker is the first column
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", 'wb') as f:
        pickle.dump(tickers, f)

    print(tickers)
    return tickers


filepath = "F:/data/sp500"


def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        save_SP500_tickers()

    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    if not os.path.exists("F:/data/sp500"):
        os.makedirs("F:/data/sp500")

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2016, 12, 31)

    for ticker in tickers:
        if not os.path.exists("F:/data/sp500/{}.csv".format(ticker)):
            try:
                df = pdr.DataReader(ticker, 'yahoo', start, end)
                df.to_csv("F:/data/sp500/{}.csv".format(ticker))
            except:
                print("Couldn't find", ticker)


get_data_from_yahoo(True)

def compile_data():
    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv(filepath + "/{}.csv".format(ticker))
            df.set_index('Date', inplace=True)
            df.rename(columns={'Adj Close' : ticker}, inplace=True)
            # default is axis 0, but if you use default then the drop crashes
            # hence you enter 1 which is the column you want to drop
            df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                # so that no dates are dropped
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)
        except:
            print(ticker,"file not found")

    print(main_df.head())
    main_df.to_csv(('sp500_joined_closes.csv'))


all_data = 'sp500_joined_closes.csv'
def visualize_data():
    df = pd.read_csv(all_data)
    df_corr = df.corr()
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # Rd = negative Yl = neutral Gn = positive
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()


def process_data(ticker):
    # predict about next 7 days
    hm_days = 7
    df = pd.read_csv(all_data)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        # the -i will give the future 1 day data
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    # if stock changes by 2% in 7 days then sell /buy
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data(ticker)
    hm_days = 7
    # predict the buy sell or hold for the next 7 days and store it in a map
    # buy_sell_hold returns a 1, or -1 if ANY of the 7 inputs exceeds requirement,
    # not a value for each input. So it returns one output for n inputs.﻿
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)]))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    # Just to see how many buy, sell or hold instructions from the model
    print("Data spread:", Counter(str_vals))
    df.fillna(0, inplace=True)
    # replace infinite changes in prices with np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # we have to be very accurate with what features to be sent and not send the columns for values to be predicted
    # normalizing the values
    df_vals = df[[t for t in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], np.nan)
    df_vals.fillna(0, inplace=True)

    x = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return x, y, df