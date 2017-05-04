import pickle
import time
import datetime as dt
import os
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import copy
from numpy import genfromtxt
#
# def getPSAR(df, ini_accleration_factor, max_accleration_factor):
# # extreme point is high of low (it gets corrected as we move to next value)
#     df.exteme_point.iloc[0] = df['Low'].iloc[0]
#     df.trend.iloc[0] = 'Bear'
#     # we started extremepoint as low hence the psar will be high
#     df['PSAR'].iloc[0] = df['High'].iloc[0]
#     df.acceleration.iloc[0] = ini_accleration_factor
#     df.psar_ep_af = (df['PSAR'].iloc[0] - df.exteme_point.iloc[0]) * df.acceleration.iloc[0]
#     # IF(L5=”falling”,MAX(K5-I5,C5,C4),IF(L5=”rising”,MIN(K5-I5,D5,D4),””))
#
#     df['PSAR'][1:] = np.where(df.trend.shift(-1) == 'Bear', np.max)
#
#     psar = df['Low']
#     ep = df['High']
#     af = af_ini
#     ep_psar = ep - psar
#     ep_psar_acc = af * ep_psar
#     trend = np.where(psar.lt(df['High']), 'bull', np.where(psar.gt(df['Low']), 'bear', ''))
#     # IF(AND(L4=”bull”,G4+K4>E5),H4,IF(AND(L4=”bear”,G4+K4<D5),H4,G4+K4))
#     psar[1:] = np.where((trend.shit(-1) == 'bull' and (psar.shift(-1) + ep_psar_acc.shift(-1)) > df['Low'])
#                         or (trend.shit(-1) == 'bear' and (psar.shift(-1) + ep_psar_acc.shift(-1)) < df['High']), ep.shift(-1), psar.shift(-1) + ep_psar_acc)
#     #  H5 =IF(AND(L5=”bull”,D5>H4),D5,IF(AND(L5=”bull”,D5<=H4),H4,IF(AND(L5=”bear”,E5<H4),E5,IF(AND(


def getRSI(df, period):
    delta = df.diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = pd.rolling_mean(up, period)
    roll_down = pd.rolling_mean(down.abs(), period)
    df['RSI'] = 100 - (100 / (1 + roll_up / roll_down))
    return df['RSI']


def getBreadthThrust(df, period):
    return df.ewm(span=period, adjust=False).mean()


def getBBands(df, window_size, num_of_std):
    rolling_mean = df.rolling(window=window_size).mean()
    rolling_std = df.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return rolling_mean, upper_band, lower_band


def getMACD(df, fastPeriod, slowPeriod, signPeriod):
    df.slow = getWildersEMA(df, slowPeriod)
    df.fast = getWildersEMA(df, fastPeriod)
    df['MACD'] = df.fast - df.slow
    df['MACD'].dropna(inplace=True)
    df['MACD_sign'] = getWildersEMA(df['MACD'], signPeriod)
    df['MACD_diff'] = df['MACD'] - df['MACD_sign']
    return df['MACD'], df['MACD_sign'], df['MACD_diff']


def getADX(df, period):
    df.moveUp = df.High - df.High.shift(1)
    df.moveDown = df.Low.shift(1) - df.Low
    df['PDM'] = np.where(df.moveUp.gt(df.moveDown) & df.moveUp.gt(0), df.moveUp, 0)
    df['NDM'] = np.where(df.moveDown.gt(df.moveUp) & df.moveDown.gt(0), df.moveDown, 0)

    # did df['diffHL'] instead of df.diffHL as the need column names for TR calculation
    df['diffHL'] = df['High'] - df['Low']
    df['diffHCy'] = abs(df['High'] - df['Close'].shift(1))
    df['diffLCy'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['diffHL', 'diffHCy', 'diffLCy']].apply(max, axis=1)
    # this gives EWMA but it is not Wilder's
    # df['expPosDM'] = df['PDM'].ewm(span=period, ignore_na=True, adjust=False).mean()
    df.PosDM = getWildersEMA(df['PDM'], period)
    # AttributeError: 'numpy.ndarray' object has no attribute 'rolling'
    # if df.NDM instead of df['NDM']
    df.NegDM = getWildersEMA(df['NDM'], period)
    df.ATR = getWildersEMA(df['TR'], period)

    # need to dropna as the wilderEMA returns starting values as NaN
    df.dropna(inplace=True)
    df.PDI = df.PosDM / df.ATR
    df.NDI = df.NegDM / df.ATR
    df['DX'] = abs(df.PDI - df.NDI) / (df.PDI + df.NDI)
    df.dropna(inplace=True)
    df['ADX'] = 100 * getWildersEMA(df['DX'], period)
    return df['ADX']


def getWildersEMA(df, period):
    ema = copy.deepcopy(df)
    ema[0: period] = np.nan
    ema[period - 1] = df.rolling(window=period).mean()[period - 1]
    # fastest way to get number of rows in a df is len(df.index)
    for i in range(period, len(df.index)):
        # Wilder EMA formula = price today * K + EMA yesterday (1-K) where K =1/N
        ema[i] = df[i] / period + ((period - 1) * ema[i - 1]) / period
    return ema


def compile_data():
    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    data = {}

    for count, ticker in enumerate(tickers):
        try:
            filepath = "F:/data/sp500/{}.csv".format(ticker)
            df = pd.read_csv(filepath)
            df.set_index('Date', inplace=True)
            df.to_csv('aapl2.csv')
            df.index = pd.to_datetime(df.index)
            ratio = df["Close"] / df["Adj Close"]
            df["Open"] = df["Open"] / ratio
            df["High"] = df["High"] / ratio
            df["Low"] = df["Low"] / ratio
            df["Close"] = df["Adj Close"]

            if ticker == 'AAPL':
                df.to_csv('aapl.csv')

            df.drop(["Adj Close"], axis=1, inplace=True)

            df['MACD'], df['MACD_sign'], df['MACD_diff'] = getMACD(copy.deepcopy(df['Close']), 12, 26, 9)
            df['ADX'] = getADX(copy.deepcopy(df), 14)
            df['BBMean'], df['BBUp'], df['BBDown'] = getBBands(copy.deepcopy(df['Close']), 20, 2)
            #df['Breadth_Thrust'] = getBreadthThrust(copy.deepcopy(df['Close']),10)
            df['RSI'] = getRSI(df['Close'], 14)
            df.dropna(inplace=True)

            df.drop(["Open", "High", "Low"], axis=1, inplace=True)
            data[ticker] = df

            if ticker == 'AAP':
                print(df.tail())
        except Exception as e:
            print(e)

    panel = pd.Panel(data)
    print(panel['AAP'].head())
    panel.to_pickle('stocks.pickle')


compile_data()
