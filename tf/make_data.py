import pickle
import talib
import pandas as pd

import numpy as np
import copy


def compile_data():
    with open("../sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    data = {}

    for count, ticker in enumerate(tickers):
        try:
            filepath = "F:/data/sp500/{}.csv".format(ticker)
            df = pd.read_csv(filepath)
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
            ratio = df["Close"] / df["Adj Close"]
            df["Open"] = df["Open"] / ratio
            df["High"] = df["High"] / ratio
            df["Low"] = df["Low"] / ratio
            df["Volume"] = df["Volume"] / ratio
            df["Close"] = df["Adj Close"]

            df.drop(["Adj Close"], axis=1, inplace=True)

            df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, 10)
            df['RSI'] = talib.RSI(df['Close'].values, 14)
            # APO(close, fastperiod=12, slowperiod=26, matype=0)
            df['APO'] = talib.APO(df['Close'].values, 12, 26, 0)

            # CCI(high, low, close, timeperiod=14)
            df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, 14)

            # WILLR(high, low, close, timeperiod=14)
            df['WILLR'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, 14)

            # TRIX(close, timeperiod=30)
            df['TRIX'] = talib.TRIX(df['Close'].values, 30)

            # CMO(close, timeperiod=14)
            df['CMO'] = talib.CMO(df['Close'].values, 14)

            # AROONOSC(high, low, timeperiod=14)
            df['AROONOSC'] = talib.AROONOSC(df['High'].values, df['Low'].values, 14)


            df.dropna(inplace=True)

            df.drop(["Open", "High", "Low", "Volume"], axis=1, inplace=True)
            data[ticker] = df

            if ticker == 'AAP':
                print(df.tail(30))
        except Exception as e:
            print(e)

    panel = pd.Panel(data)
    panel.to_pickle('../stocks.pickle')


compile_data()
