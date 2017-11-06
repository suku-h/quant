import talib
from objects import Indicator, ReqParam
from data import getStocks
from util import removeNaN
import numpy as np


def getColumnKey(indicator):
    # there is no switch case in python hence you use a dictionary
    switch = {
        'ADX': 2,
        'ADXR': 2,
        'APO': 4,
        'AROONOSC': 1,
        'BOP': 3,
        'CCI': 2,
        'CMO': 0,
        'DX': 2,
        'MFI': 5,
        'MINUS_DI': 2,
        'MINUS_DM': 2,
        'MOM': 0,
        'PLUS_DI': 2,
        'PLUS_DM': 2,
        'PPO': 4,
        'ROC': 0,
        'ROCP': 0,
        'ROCR': 0,
        'ROCR100': 0,
        'RSI': 0,
        'TRIX': 0,
        'ULTOSC': 6,
        'WILLR': 2,
        'AD': 5,
        'ADOSC': 7,
        'OBV': 8,
        'ATR': 2,
        'NATR': 2,
        'TRANGE': 9
    }

    return switch.get(indicator)


def getIndicatorValues(indicator: Indicator, data):
    needs = getColumnKey(indicator.name)
    indicatorMap = {}

    if needs is None:
        raise ValueError('indicator ' + indicator.name + ' is not part of talib')

    _, stocks = getStocks()
    for count, ticker in enumerate(stocks):
        df = data.get(ticker)

        if needs == 0:
            df[indicator.name] = getattr(talib, indicator.name)(df['Close'].values, timeperiod=indicator.period)
        elif needs == 1:
            df[indicator.name] = getattr(talib, indicator.name)(df['High'].values, df['Low'].values, timeperiod=indicator.period)
        elif needs == 2:
            df[indicator.name] = getattr(talib, indicator.name)(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=indicator.period)
        elif needs == 3:
            df[indicator.name] = getattr(talib, indicator.name)(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
        elif needs == 4:
            df[indicator.name] = getattr(talib, indicator.name)(df['Close'].values, fastperiod=indicator.fastperiod, slowperiod=indicator.slowperiod, matype=0)
        elif needs == 5:
            df[indicator.name] = getattr(talib, indicator.name)(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values, timeperiod=indicator.period)
        elif needs == 6:
            df[indicator.name] = getattr(talib, indicator.name)(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values, timeperiod1=indicator.timeperiod1, timeperiod2=indicator.timeperiod2, timeperiod3=indicator.timeperiod3)
        elif needs == 7:
            df[indicator.name] = getattr(talib, indicator.name)(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values, fastperiod=indicator.fastperiod, slowperiod=indicator.slowperiod)
        elif needs == 8:
            df[indicator.name] = getattr(talib, indicator.name)(df['Close'].values, df['Volume'].values)
        elif needs == 9:
            df[indicator.name] = getattr(talib, indicator.name)(df['High'].values, df['Low'].values, df['Close'].values)

        if needs == 0 or needs == 1 or needs == 2 or needs == 5:
            indicatorMap[ticker + '_' + str(indicator.period)] = df[indicator.name]
        elif needs == 4 or needs == 7:
            indicatorMap[ticker + '_' + str(indicator.fastperiod) + '_' + str(indicator.slowperiod)] = df[indicator.name]
        elif needs == 6:
            indicatorMap[ticker + '_' + str(indicator.timeperiod1) + '_' + str(indicator.timeperiod2) + '_' + str(indicator.timeperiod3)] = df[indicator.name]
        elif needs == 3 or needs == 8 or needs == 9:
            indicatorMap[ticker] = df[indicator.name]

    return indicatorMap


def getRequirementData(req: ReqParam, data):
    reqMap = {}

    _, stocks = getStocks()
    for count, ticker in enumerate(stocks):
        df = data.get(ticker)

        if req.condition == 1:
            if req.indicatorPeriod is None:
                raise ValueError('Need to provide indicator period')
            df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, req.indicatorPeriod)
            reqMap[ticker + '_' + str(req.indicatorPeriod)] = removeNaN(df['ADX'].values)
        if req.condition == 2:
            if req.fastPeriod is None or req.slowPeriod is None:
                raise ValueError('Need to provide fastperiod or slowperiod')
            df['chaikin'] = talib.ADOSC(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values, req.fastPeriod, req.slowPeriod)
            reqMap[ticker + '_' + str(req.fastPeriod) + '_' + str(req.slowPeriod)] = removeNaN(df['chaikin'].values)
        if req.condition == 3:
            if req.indicatorPeriod is None or req.fastPeriod is None or req.slowPeriod is None:
                raise ValueError('Need to provide indicatorPeriod, fastPeriod or slowPeriod')
            _, __, df['MACDHist'] = talib.MACD(df['Close'].values, fastperiod=req.fastPeriod, slowperiod=req.slowPeriod, signalperiod=req.indicatorPeriod)
            reqMap[ticker + '_' + str(req.indicatorPeriod) + '_' + str(req.fastPeriod) + '_' + str(req.slowPeriod)] = removeNaN(df['MACDHist'].values)
        if req.condition == 4:
            if req.rollingPeriod is None:
                raise ValueError('Need to provide rollingPeriod')
            df['vol'] = np.where(df['Volume'] > df['Volume'].rolling(req.rollingPeriod).mean())
            reqMap[ticker + '_' + str(req.rollingPeriod)] = df['vol'].values

    return reqMap
