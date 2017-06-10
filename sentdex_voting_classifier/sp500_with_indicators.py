import pickle
import time
import datetime as dt
import copy
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import copy
from numpy import genfromtxt

def getPSAR(df, ini_acceleration_factor, max_acceleration_factor):
    df['PSAR'] = df['Low']
    df['EP'] = df['Low']
    df['acceleration'] = ini_acceleration_factor
    df['direction'] = 'bull'
    df['temp_psar'] = df['PSAR']

    for i in range(1, len(df)):
        if df['direction'].iloc[i-1] == 'bull' and df['EP'].iloc[i-1] < df['High'].iloc[i]:
            af = np.minimum(df['acceleration'].iloc[i - 1] + ini_acceleration_factor, max_acceleration_factor)
            df['EP'].iloc[i] = df['High'].iloc[i]
        elif df['direction'].iloc[i-1] == 'bear' and df['EP'].iloc[i-1] > df['Low'].iloc[i]:
            af = np.minimum(df['acceleration'].iloc[i - 1] + ini_acceleration_factor, max_acceleration_factor)
            df['EP'].iloc[i] = df['Low'].iloc[i]
        else:
            af = df['acceleration'].iloc[i - 1]

        df['temp_psar'].iloc[i] = df['PSAR'].iloc[i-1] + af * (df['EP'].iloc[i] - df['PSAR'].iloc[i-1])

        if df['direction'].iloc[i-1] == 'bull' and df['Low'].iloc[i] < df['temp_psar'].iloc[i]:
            df['direction'].iloc[i] = 'bear'
            df['EP'].iloc[i] = df['Low'].iloc[i]
        elif df['direction'].iloc[i-1] == 'bear' and df['High'].iloc[i] > df['temp_psar'].iloc[i]:
            df['direction'].iloc[i] = 'bull'
            df['EP'].iloc[i] = df['High'].iloc[i]
        else:
            df['direction'].iloc[i] = df['direction'].iloc[i-1]

        if df['direction'].iloc[i] == df['direction'].iloc[i-1]:
            if (df['direction'].iloc[i] == 'bull' and df['EP'].iloc[i] < df['High'].iloc[i]) or \
                    (df['direction'].iloc[i] == 'bear' and df['EP'].iloc[i] > df['Low'].iloc[i]):
                df['acceleration'].iloc[i] = np.minimum(df['acceleration'].iloc[i - 1] + ini_acceleration_factor, max_acceleration_factor)
            else:
                df['acceleration'].iloc[i] = df['acceleration'].iloc[i - 1]
        else:
            df['acceleration'].iloc[i] = ini_acceleration_factor

        if df['acceleration'].iloc[i] == ini_acceleration_factor and df['direction'].iloc[i] == 'bear':
            df['PSAR'].iloc[i] = df['High'].iloc[i-1]
        elif df['acceleration'].iloc[i] == ini_acceleration_factor and df['direction'].iloc[i] == 'bull':
            df['PSAR'].iloc[i] = df['Low'].iloc[i-1]
        else:
            df['PSAR'].iloc[i] = df['PSAR'].iloc[i - 1] + df['acceleration'].iloc[i] * (df['EP'].iloc[i] - df['PSAR'].iloc[i - 1])


    print(df)
    return df['PSAR']


def getWilliamPercentR(df, period):
    df['Highest_High'] = df['High'].rolling(window=period).max()
    df['Lowest_Low'] = df['Low'].rolling(window=period).min()
    df['William %R'] = 100 * (df['Highest_High'] - df['Close'])/(df['Highest_High'] - df['Lowest_Low'])
    return df['William %R']


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


def getChaikinOscillator(df, short_period, long_period):
    df['Money_Flow_Multiplier'] = (2 * df['Close'] - df['Low'] - df['High'])/(df['High'] - df['Low'])
    df['Money_Flow_Volume'] = df['Money_Flow_Multiplier'] * df['Volume'].rolling(window=long_period).sum()
    df['ADL'] = df['ADL'].shift(1) + df['Money_Flow_Volume']
    df['Chaikin'] = pd.ewma(df['ADL'], span=short_period, freq="D") - pd.ewma(df['ADL'], span=long_period, freq="D")
    return df['Chaikin']


def compile_data():
    with open("../sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    data = {}

    for count, ticker in enumerate(tickers):
        try:
            filepath = "F:/data/sp500/{}.csv".format(ticker)
            df = pd.read_csv(filepath)
            df.set_index('Date', inplace=True)
            df.to_csv('aap.csv')
            df.index = pd.to_datetime(df.index)
            ratio = df["Close"] / df["Adj Close"]
            df["Open"] = df["Open"] / ratio
            df["High"] = df["High"] / ratio
            df["Low"] = df["Low"] / ratio
            df["Close"] = df["Adj Close"]

            if ticker == 'AAP':
                df.to_csv('aap.csv')

            df.drop(["Adj Close"], axis=1, inplace=True)

            # df['MACD'], df['MACD_sign'], df['MACD_diff'] = getMACD(copy.deepcopy(df['Close']), 12, 26, 9)
            # df['ADX'] = getADX(copy.deepcopy(df), 14)
            # df['BBMean'], df['BBUp'], df['BBDown'] = getBBands(copy.deepcopy(df['Close']), 20, 2)
            # #df['Breadth_Thrust'] = getBreadthThrust(copy.deepcopy(df['Close']),10)
            # df['RSI'] = getRSI(df['Close'], 14)

            # df['PSAR'] = getPSAR(df, 0.02, 0.2)
            df['William %R'] = getWilliamPercentR(df, 10)
            df['Chaikin'] = getChaikinOscillator(df, 3, 10)
            df.dropna(inplace=True)

            df.drop(["Open", "High", "Low"], axis=1, inplace=True)
            data[ticker] = df

            if ticker == 'AAP':
                print(df.tail(100))
        except Exception as e:
            print(e)

    panel = pd.Panel(data)
    print(panel['AAPL'].tail(100))
    panel.to_pickle('stocks.pickle')


compile_data()
