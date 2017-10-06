import talib
import pandas as pd
import numpy as np
import csv
import math
import glob
from datetime import datetime
from copy import deepcopy
import operator


def analyseStrategy(res):
    if np.isnan(res[9]) or res[9] == 0 or np.isnan(res[3]) or res[3] < 0 or np.isnan(res[6]):
        return 0

    trades = 5000 * (res[9] / res[0])
    sellPrice = 100
    for i in range(math.floor(trades)):
        sellPrice *= (1 + res[3] / 100)

    sellPrice *= (1 + res[3] * (trades - math.floor(trades)) / 100)

    return math.floor(sellPrice) - 100


def backTestStrategy(trades):
    buyDict = {}
    iniVal = 250000
    money = deepcopy(iniVal)
    nav = deepcopy(iniVal)
    for key, dayTrade in sorted(trades.items(), key=lambda t: datetime.strptime(t[0], "%Y-%m-%d").timestamp()):
        for stock in dayTrade.sells:
            # .4% loss due to brockerage and taxes
            money += buyDict.get(stock.symbol)[0] * stock.wap * 0.996
            nav += buyDict.get(stock.symbol)[0] * (stock.wap * 0.996- buyDict.get(stock.symbol)[1])
            buyDict[stock.symbol] = (0, 0)

        # // gives int division so no decimal
        max_per_stock = nav // 15
        sortedBuys = sorted(dayTrade.buys, key=operator.attrgetter('indicatorVal'))
        for s in sortedBuys:
            amount = min(max_per_stock, money)
            buyDict[s.symbol] = (amount // s.wap, s.wap)
            money -= amount - amount % s.wap


    return nav - iniVal



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


def getIndicatorValues(indicator, needs, df, period, fastperiod=0, slowperiod=0, timeperiod1=7, timeperiod2=14,
                       timeperiod3=28):
    if needs == 0:
        return getattr(talib, indicator)(df['Close'].values, timeperiod=period)
    elif needs == 1:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, timeperiod=period)
    elif needs == 2:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=period)
    elif needs == 3:
        return getattr(talib, indicator)(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
    elif needs == 4:
        return getattr(talib, indicator)(df['Close'].values, fastperiod=fastperiod, slowperiod=slowperiod, matype=0)
    elif needs == 5:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values,
                                         timeperiod=period)
    elif needs == 6:
        return getattr(talib, indicator)(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values,
                                         timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
    elif needs == 7:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values,
                                         fastperiod=fastperiod, slowperiod=slowperiod)
    elif needs == 8:
        return getattr(talib, indicator)(df['Close'].values, df['Volume'].values)
    elif needs == 9:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values)


def analyze_indicator(indicator, file, max_buy_val, min_sell_val, period, fastperiod=0, slowperiod=0, timeperiod1=7,
                      timeperiod2=14, timeperiod3=28):
    total_res = np.zeros(15)
    data = pd.read_csv(file)
    stocks = data['Symbol'].values
    for i in range(len(stocks)):
        if stocks[i] == 'FAGBEARING':
            stocks[i] = 'SCHAEFFLER'

    needs = getColumnKey(indicator)

    trades = {}

    # without count the ticker name is incorrect
    for count, ticker in enumerate(stocks):
        total_df = pd.read_csv("F:/data/nse500/selected/{}.csv".format(ticker))
        total_df.reset_index(inplace=True)
        df = total_df[['Symbol', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'VWAP']]
        df.set_index(['Symbol', 'Date'], inplace=True)

        # make Volume as double as talib needs it as double/float
        df['Volume'] = df['Volume'].astype(float)

        df[indicator] = getIndicatorValues(indicator, needs, df, period, fastperiod=fastperiod, slowperiod=slowperiod,
                                           timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
        df.dropna(axis=0, inplace=True)

        df['Res'] = np.where(df[indicator] < max_buy_val, 1, np.where(df[indicator] > min_sell_val, -1, 0))
        df['maV'] = df['Volume'].rolling(100).mean()
        df['anaV'] = np.where(df['Volume'] > 1.25 * df['maV'], 1, 0)

        try:
            _, __, df['MACDHist'] = talib.MACD(df['Close'].values)
        except:
            print(ticker)

        df.dropna(axis=0, inplace=True)

        op = df['Res'].values
        indicatorVals = df[indicator].values
        # this creates a numpy array of index tuples (ticker, date)
        # the date is the 2nd param & hence only date is referred as dates[i][1]
        dates = df.index.tolist()
        vals = df['VWAP'].values
        amt = np.zeros(len(df['Res']))
        gain = np.zeros(len(df['Res']))
        day_diff = np.zeros(len(df['Res'])).astype(int)
        sell_days = np.zeros(len(df['Res'])).astype(int)
        anaV = df['anaV'].values
        hist = df['MACDHist'].values

        buy_signal = 0
        sell_signal = 0
        closePosition = -1

        # don't check volume as volume already part of the calculation
        lastBought = ''
        if needs == 5 or needs == 7 or needs == 8:
            for i in range(len(op) - 1):
                if op[i] > buy_signal and hist[i] < 0 and i > closePosition:
                    addTrade(trades, ticker, dates[i + 1][1], vals[i + 1], indicatorVals[i], True)
                    for j in range(i + 1, len(op) - 1):
                        if op[j] < sell_signal and hist[j] > 0:
                            addTrade(trades, ticker, dates[j + 1][1], vals[j + 1], indicatorVals[j], False)
                            amt[j] = vals[j] - vals[i]
                            gain[j] = (vals[j] - vals[i]) * 100 / vals[i]
                            day_diff[j] = j - i
                            closePosition = j
                            break
        else:
            for i in range(len(op) - 1):
                if op[i] > buy_signal and hist[i] < 0 and anaV[i] == 1 and i > closePosition:
                    addTrade(trades, ticker, dates[i + 1][1], vals[i + 1], indicatorVals[i], True)
                    noBuys = False
                    for j in range(i + 1, len(op) - 1):
                        if op[j] < sell_signal and hist[j] > 0 and anaV[j] == 1:
                            addTrade(trades, ticker, dates[j + 1][1], vals[j + 1], indicatorVals[j], False)
                            amt[j] = vals[j] - vals[i]
                            gain[j] = (vals[j] - vals[i]) * 100 / vals[i]
                            day_diff[j] = j - i
                            closePosition = j
                            break

                        if j == len(op) - 2:
                            noBuys = True

                    if noBuys:
                        break

        df['Amt'] = amt[:]
        df['Gain'] = gain[:]
        df['Day_Diff'] = day_diff[:]
        df['Sell_Days'] = sell_days[:]

        total_res[0] += len(df['Gain'])
        total_res[1] += len(df['Gain'][df['Gain'] > 0])
        total_res[2] += len(df['Gain'][df['Gain'] < 0])
        total_res[3] += df['Gain'][(df['Gain'] > 0) | (df['Gain'] < 0)].sum()
        total_res[4] += df['Gain'][df['Gain'] > 0].sum()
        total_res[5] += df['Gain'][df['Gain'] < 0].sum()
        total_res[6] += df['Day_Diff'][(df['Gain'] > 0) | (df['Gain'] < 0)].sum()
        total_res[7] += df['Day_Diff'][df['Gain'] > 0].sum()
        total_res[8] += df['Day_Diff'][df['Gain'] < 0].sum()
        total_res[9] += len(df['Gain'][(df['Gain'] > 0) | (df['Gain'] < 0)])
        total_res[10] += len(df['Sell_Days'][df['Sell_Days'] > 0])

    result = np.zeros(12)
    result[0] = total_res[0]
    result[1] = total_res[1]
    result[2] = total_res[2]
    result[3] = total_res[3] / total_res[9]
    result[4] = total_res[4] / total_res[1]
    result[5] = total_res[5] / total_res[2]
    result[6] = total_res[6] / total_res[9]
    result[7] = total_res[7] / total_res[1]
    result[8] = total_res[8] / total_res[2]
    result[9] = total_res[9]
    result[10] = analyseStrategy(result)

    result[11] = backTestStrategy(trades)

    print('Gain', result[10])
    print('NAV', result[11])
    print('\n')

    group = file[32: len(file) - 8]
    row = [indicator, group, max_buy_val, min_sell_val, period]

    if needs == 4 or needs == 7:
        row.append(fastperiod)
        row.append(slowperiod)
        row.append("-")
        row.append("-")
        row.append("-")
    elif needs == 6:
        row.append("-")
        row.append("-")
        row.append(timeperiod1)
        row.append(timeperiod2)
        row.append(timeperiod3)
    else:
        for j in range(5):
            row.append("-")

    # check for diff append and extend https://stackoverflow.com/a/252711/5512020
    row.extend([element for element in result])
    # without newline = '' , a row is skipped in csv
    # with open('analysis.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(row)
    # f.close()


def addTrade(trades, ticker, date, val, indicatorVal, isBuy):
    s = Stock()
    s.symbol = ticker
    s.indicatorVal = indicatorVal
    s.wap = val

    if trades.get(date) is not None:
        ndt = trades.get(date)
        if isBuy:
            ndt.buys.append(s)
        else:
            ndt.sells.append(s)
        trades[date] = ndt

    else:
        dt = DayTrades()
        dt.date = date
        if isBuy:
            dt.buys.append(s)
        else:
            dt.sells.append(s)
        trades[date] = dt


class Stock():
    symbol = ""
    indicatorVal = 0
    wap = 0


class DayTrades():
    date = 0
    moneyIn = 0
    buys = []
    sells = []
    moneyLeft = 0

    # without init the object will have previous values whenever initialized
    def __init__(self):
        super().__init__()
        self.date = 0
        self.moneyIn = 0
        self.buys = []
        self.sells = []
        self.moneyLeft = 0


def runSinglePeriodIndicatorAnalyses(indicator, maxBuyVals, minSellVals, periods):
    count = 0
    path = r'F:\data\nse500\indices'
    allFiles = glob.glob(path + "\*.csv")
    for i in range(len(maxBuyVals)):
        for j in range(len(minSellVals)):
            for k in range(len(periods)):
                for f in allFiles:
                    try:
                        count += 1
                        print(indicator + " Analyses: ", f[32: len(f) - 8], count, maxBuyVals[i], minSellVals[j],
                              periods[k])
                        analyze_indicator(indicator=indicator, file=f, max_buy_val=maxBuyVals[i],
                                          min_sell_val=minSellVals[j], period=periods[k])
                    except Exception as e:
                        print(e)
                        print(indicator)


def analyseCMO():
    maxBuyVals = np.array([-35, -40, -45, -50, -55])
    minSellVals = np.array([35, 40, 45, 50, 55])
    periods = np.array([10, 12, 14, 16, 18, 20])
    runSinglePeriodIndicatorAnalyses('CMO', maxBuyVals, minSellVals, periods)


def analyseRSI():
    maxBuyVals = np.array([20, 25, 30, 35, 40])
    minSellVals = np.array([55, 60, 65, 70, 75])
    periods = np.array([10, 12, 14, 16, 18, 20])
    runSinglePeriodIndicatorAnalyses('RSI', maxBuyVals, minSellVals, periods)


def analyseAROONOSC():
    maxBuyVals = np.array([-40, -50, -60])
    minSellVals = np.array([40, 50, 60])
    periods = np.array([10, 12, 14, 16, 18, 20])
    runSinglePeriodIndicatorAnalyses('AROONOSC', maxBuyVals, minSellVals, periods)


def analyseCCI():
    maxBuyVals = np.array([-90, -95, -100, -105, -110])
    minSellVals = np.array([90, 95, 100, 105, 100])
    periods = np.array([14, 16, 18, 20, 22, 24])
    runSinglePeriodIndicatorAnalyses('CCI', maxBuyVals, minSellVals, periods)


def analyseMFI():
    maxBuyVals = np.array([15, 20, 25, 30])
    minSellVals = np.array([70, 75, 80, 85])
    periods = np.array([10, 12, 14, 16, 18, 20])
    runSinglePeriodIndicatorAnalyses('MFI', maxBuyVals, minSellVals, periods)


# Check the answer https://stackoverflow.com/a/20627316/5512020
# to prevent the warning: A value is trying to be set on a copy of a slice from a DataFrame.
pd.options.mode.chained_assignment = None  # default='warn'

# analyseCMO()
# analyseRSI()
# analyseAROONOSC()
# analyseCCI()
# analyseMFI()


# def panalyseMFI():
#     maxBuyVals = np.array([30])
#     minSellVals = np.array([70])
#     periods = np.array([10, 12, 14, 16])
#     runSinglePeriodIndicatorAnalyses('MFI', maxBuyVals, minSellVals, periods)
#
#
# panalyseMFI()

def p(indicator, maxBuyVals, minSellVals, periods):
    count = 0
    path = r'F:\data\nse500\indices'
    allFiles = glob.glob(path + "\*.csv")
    for i in range(len(maxBuyVals)):
        for j in range(len(minSellVals)):
            for k in range(len(periods)):
                for f in allFiles:
                    if f.find("fmcg") > 0:
                        analyze_indicator(indicator=indicator, file=f, max_buy_val=maxBuyVals[i],
                                          min_sell_val=minSellVals[j], period=periods[k])

def aanalyseCMO():
    maxBuyVals = np.array([-50])
    minSellVals = np.array([55])
    periods = np.array([20])
    p('CMO', maxBuyVals, minSellVals, periods)


aanalyseCMO()