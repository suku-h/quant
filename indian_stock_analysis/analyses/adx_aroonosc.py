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


def backTestStrategy(trades, isNifty500):
    buyDict = {}
    iniVal = 250000
    money = deepcopy(iniVal)
    nav = deepcopy(iniVal)
    for key, dayTrade in sorted(trades.items(), key=lambda t: datetime.strptime(t[0], "%Y-%m-%d").timestamp()):
        for stock in dayTrade.sells:
            # .4% loss due to brockerage and taxes
            money += buyDict.get(stock.symbol)[0] * stock.wap * 0.996
            nav += buyDict.get(stock.symbol)[0] * (stock.wap * 0.996 - buyDict.get(stock.symbol)[1])
            buyDict[stock.symbol] = (0, 0)

        # // gives int division so no decimal
        max_stocks = 15 if isNifty500 else 4
        max_per_stock = nav // max_stocks
        sortedBuys = sorted(dayTrade.buys, key=operator.attrgetter('indicatorVal'))
        for s in sortedBuys:
            amount = min(max_per_stock, money)
            buyDict[s.symbol] = (amount // s.wap, s.wap)
            money -= amount - amount % s.wap

    return nav - iniVal


def analyze_indicator(file, max_buy_val, min_sell_val, aroon_period, adx_period, min_adx):
    total_res = np.zeros(15)
    data = pd.read_csv(file)
    stocks = data['Symbol'].values
    for i in range(len(stocks)):
        if stocks[i] == 'FAGBEARING':
            stocks[i] = 'SCHAEFFLER'

    trades = {}

    indicator = 'AROONOSC'
    # without count the ticker name is incorrect
    for count, ticker in enumerate(stocks):
        total_df = pd.read_csv("F:/data/nse500/selected/{}.csv".format(ticker))
        total_df.reset_index(inplace=True)
        df = total_df[['Symbol', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'VWAP']]
        df.set_index(['Symbol', 'Date'], inplace=True)

        # make Volume as double as talib needs it as double/float
        df['Volume'] = df['Volume'].astype(float)

        df[indicator] = talib.AROONOSC(df['High'].values, df['Low'].values, aroon_period)
        df['Res'] = np.where((df[indicator] <= max_buy_val) & (df[indicator].shift(1) <= df[indicator]) & (
            df[indicator].shift(1) < max_buy_val), 1, np.where((df[indicator] >= min_sell_val) & (
            df[indicator].shift(1) >= df[indicator]) & (df[indicator].shift(1) > max_buy_val), -1, 0))
        df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, adx_period)

        df.dropna(axis=0, inplace=True)

        op = df['Res'].values
        indicatorVals = df[indicator].values
        adx = df['ADX'].values

        # this creates a numpy array of index tuples (ticker, date)
        # the date is the 2nd param & hence only date is referred as dates[i][1]
        dates = df.index.tolist()
        vals = df['VWAP'].values
        amt = np.zeros(len(df['Res']))
        gain = np.zeros(len(df['Res']))
        day_diff = np.zeros(len(df['Res'])).astype(int)
        sell_days = np.zeros(len(df['Res'])).astype(int)

        buy_signal = 0
        sell_signal = 0
        closePosition = -1

        # don't check volume as volume already part of the calculation
        for i in range(4, len(op) - 1):
            if op[i] > buy_signal and adx[i] > min_adx and i > closePosition:
                addTrade(trades, ticker, dates[i + 1][1], vals[i + 1], indicatorVals[i], True)
                noBuys = False
                for j in range(i + 1, len(op) - 1):
                    if op[j] < sell_signal and adx[i] > min_adx:
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

    group = file[32: len(file) - 8]

    result[11] = backTestStrategy(trades, group == '500')

    print('Gain', result[10])
    print('NAV', result[11])
    print('\n')

    row = [group, max_buy_val, min_sell_val, aroon_period, adx_period, min_adx]


    # check for diff append and extend https://stackoverflow.com/a/252711/5512020
    row.extend([element for element in result])
    # without newline = '' , a row is skipped in csv
    with open('adxaroon_analysis.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    f.close()


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


def analyseAROONOSC():
    maxBuyVals = np.array([-40, -50, -60])
    minSellVals = np.array([40, 50, 60])
    periods = np.array([6, 8, 10, 12, 14, 16])
    adx_periods = np.array([6, 8, 10, 12, 14, 16])
    adx_mins = np.array([25, 30, 35])

    count = 0
    path = r'F:\data\nse500\indices'
    allFiles = glob.glob(path + "\*.csv")
    for i in range(len(maxBuyVals)):
        for j in range(len(minSellVals)):
            for k in range(len(periods)):
                for l in range(len(adx_periods)):
                    for m in range(len(adx_mins)):
                        for f in allFiles:
                            try:
                                count += 1
                                print("Analyses: ", f[32: len(f) - 8], count, maxBuyVals[i], minSellVals[j],
                                      periods[k], adx_periods[l], adx_mins[m])
                                analyze_indicator(file=f, max_buy_val=maxBuyVals[i], min_sell_val=minSellVals[j],
                                                  aroon_period=periods[k], adx_period=adx_periods[l],
                                                  min_adx=adx_mins[m])
                            except Exception as e:
                                print(e)
                                print(i, j, k, l, m)


pd.options.mode.chained_assignment = None

analyseAROONOSC()