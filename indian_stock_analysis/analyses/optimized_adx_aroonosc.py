import talib
import pandas as pd
import numpy as np
import csv
import glob
from datetime import datetime
from copy import deepcopy
import operator
import time

data = {}
aroonMap = {}
resMap = {}
adxMap = {}
chaikinMap = {}
histMap = {}
volMap = {}


class Stock():
    symbol = ""
    indicatorVal = 0
    wap = 0
    max_quantity = 0


class DayTrades():
    date = 0
    buys = []
    sells = []

    # without init the object will have previous values whenever initialized
    def __init__(self):
        super().__init__()
        self.date = 0
        self.buys = []
        self.sells = []


class ReqParam():
    condition = 0
    rollingPeriod = None
    indicatorPeriod = None
    fastPeriod = None
    slowPeriod = None
    adx_min = 0
    isBuy = None

    def __init__(self):
        super().__init__()
        self.condition = 0
        self.rollingPeriod = None
        self.indicatorPeriod = None
        self.fastPeriod = None
        self.slowPeriod = None
        self.adx_min = None
        self.isBuy = None

    # repr function calling return str(self.__dict__)
    # then return only keys that are not None
    def __repr__(self):
        return str({k: v for k, v in self.__dict__.items() if v is not None})


def getStocks(file='F:/data/nse500/indices/ind_nifty500list.csv'):
    data = pd.read_csv(file)
    stocks = data['Symbol'].values
    for i in range(len(stocks)):
        if stocks[i] == 'FAGBEARING':
            stocks[i] = 'SCHAEFFLER'

    return stocks


def getData():
    stocks = getStocks()
    for count, ticker in enumerate(stocks):
        total_df = pd.read_csv("F:/data/nse500/selected/{}.csv".format(ticker))
        total_df.reset_index(inplace=True)
        df = total_df[['Symbol', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'VWAP']]
        df.set_index(['Symbol', 'Date'], inplace=True)

        # make Volume as double as talib needs it as double/float
        df['Volume'] = df['Volume'].astype(float)

        data[ticker] = df


def removeNaN(arr):
    return arr[~np.isnan(arr)]


def getAROONOSC(period):
    stocks = getStocks()
    for count, ticker in enumerate(stocks):
        df = data.get(ticker)
        df['AROON'] = talib.AROONOSC(df['High'].values, df['Low'].values, period)

        # You cannot concatenate a string with an int. You would need to convert your int to a string
        aroonMap[ticker + "_" + str(period)] = df['AROON']


def getRes(max_buy_val, min_sell_val, period):
    stocks = getStocks()
    for count, ticker in enumerate(stocks):
        # df has to be initialized as a pandas dataframe otherwise
        # python treats df as an object
        df = pd.DataFrame()
        df['AROON'] = aroonMap.get(ticker + '_' + str(period))

        df['Res'] = np.where((df['AROON'] <= max_buy_val) & (df['AROON'].shift(1) <= df['AROON']) & (
            df['AROON'].shift(1) < max_buy_val), 1, np.where((df['AROON'] >= min_sell_val) & (
            df['AROON'].shift(1) >= df['AROON']) & (df['AROON'].shift(1) > max_buy_val), -1, 0))

        resMap[ticker + '_' + str(period) + '_mbv_' + str(max_buy_val) + '_msv_' + str(min_sell_val)] = removeNaN(
            df['Res'].values)


def getRequirementData(req: ReqParam):
    stocks = getStocks()
    for count, ticker in enumerate(stocks):
        df = data.get(ticker)

        if req.condition == 1:
            if req.indicatorPeriod is None:
                raise ValueError('Need to provide indicator period')
            df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, req.indicatorPeriod)
            adxMap[ticker + '_' + str(req.indicatorPeriod)] = removeNaN(df['ADX'].values)
        if req.condition == 2:
            if req.fastPeriod is None or req.slowPeriod is None:
                raise ValueError('Need to provide fastperiod or slowperiod')
            df['chaikin'] = talib.ADOSC(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values,
                                        req.fastPeriod, req.slowPeriod)
            chaikinMap[ticker + '_' + str(req.fastPeriod) + '_' + str(req.slowPeriod)] = removeNaN(df['chaikin'].values)
        if req.condition == 3:
            if req.indicatorPeriod is None or req.fastPeriod is None or req.slowPeriod is None:
                raise ValueError('Need to provide indicatorPeriod, fastPeriod or slowPeriod')
            _, __, df['MACDHist'] = talib.MACD(df['Close'].values, fastperiod=req.fastPeriod, slowperiod=req.slowPeriod,
                                               signalperiod=req.indicatorPeriod)
            histMap[ticker + '_' + str(req.indicatorPeriod) + '_' + str(req.fastPeriod) + '_' + str(req.slowPeriod)] = \
                removeNaN(df['MACDHist'].values)
        if req.condition == 4:
            if req.rollingPeriod is None:
                raise ValueError('Need to provide rollingPeriod')
            df['vol'] = np.where(df['Volume'] > df['Volume'].rolling(req.rollingPeriod).mean())
            volMap[ticker + '_' + str(req.rollingPeriod)] = df['vol'].values


def addTrade(trades, ticker, date, val, volume=0, indicatorVal=0, isBuy=True):
    s = Stock()
    s.symbol = ticker
    s.wap = val

    if isBuy:
        s.indicatorVal = indicatorVal
        s.max_quantity = volume // 10

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


def checkRequirement(req: ReqParam, position: int=None, vals: list=None) -> bool:
    if req.condition == 0:
        return True
    elif req.condition != 0 and len(vals) == 0:
        return False
    elif req.condition == 1:
        if position is None or vals is None or req.adx_min is None:
            raise ValueError('Need values for position, adx and adx_min')
        return vals[position] > req.adx_min
    elif req.condition == 2:
        if position is None or vals is None:
            raise ValueError('Need values for position and chaikin')
        return vals[position] > 0
    elif req.condition == 3:
        if position is None or vals is None or req.isBuy is None:
            raise ValueError('Need values for position, isBuy and hist')
        return vals[position] > 0 if req.isBuy else vals[position] < 0
    elif req.condition == 4:
        if position is None or vals is None:
            raise ValueError('Need values for position and vol')
        return vals[position] > 0
    elif req.condition == 5:
        if position is None or vals is None or req.isBuy is None:
            raise ValueError('Need values for position, isBuy and vals')
        return vals[position - 1] <= vals[position] if req.isBuy else vals[position - 1] >= vals[position]
    else:
        raise ValueError('Need condition needs to be between 0 & 5')


def getIndicatorValues(req: ReqParam, ticker):
    if req.condition == 0:
        return []
    elif req.condition == 1:
        return adxMap[ticker + '_' + str(req.indicatorPeriod)]
    elif req.condition == 2:
        return chaikinMap[ticker + '_' + str(req.fastPeriod) + '_' + str(req.slowPeriod)]
    elif req.condition == 3:
        return histMap[ticker + '_' + str(req.indicatorPeriod) + '_' + str(req.fastPeriod) + '_' + str(req.slowPeriod)]
    elif req.condition == 4:
        return volMap[ticker + '_' + str(req.rollingPeriod)]
    elif req.condition == 5:
        return aroonMap[ticker + '_' + str(req.indicatorPeriod)].values


def backTestStrategy(trades, isNifty500, date=None):
    buyDict = {}
    iniVal = 250000
    money = deepcopy(iniVal)
    nav = deepcopy(iniVal)

    if date is None:
        tradeSubset = trades
    else:
        # to analyse trades only for 2017, use filter
        timeStamp = datetime.strptime(date, "%Y-%m-%d").timestamp()
        tradeSubset = dict(filter(lambda x: datetime.strptime(x[0], "%Y-%m-%d").timestamp() > timeStamp, trades.items()))

    for key, dayTrade in sorted(tradeSubset.items(), key=lambda t: datetime.strptime(t[0], "%Y-%m-%d").timestamp()):
        for stock in dayTrade.sells:
            # cases where date is not None, buyDict will return none for early dates
            if buyDict.get(stock.symbol) is not None:
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
            quantity = min(amount // s.wap, s.max_quantity)
            buyDict[s.symbol] = (quantity, s.wap)
            money -= amount - amount % s.wap

    return nav - iniVal


def getMaxDataPoints(res: np.ndarray, indicatorVals: np.ndarray, r1, r2, r3, r4):
    if len(r1) == 0:
        r1 = deepcopy(res)
    if len(r2) == 0:
        r2 = deepcopy(res)
    if len(r3) == 0:
        r3 = deepcopy(res)
    if len(r4) == 0:
        r4 = deepcopy(res)

    return min(len(res), len(indicatorVals), len(r1), len(r2), len(r3), len(r4))


def getUsableDataPoints(arr, max_len):
    if len(arr) > 0:
        return arr[len(arr) - max_len:]
    else:
        return arr


def doCalculation(file, max_buy_val, min_sell_val, aroon_period, req1=ReqParam(), req2=ReqParam(), req3=ReqParam(),
                  req4=ReqParam()):
    trades = {}
    total_res = np.zeros(15)
    stocks = getStocks(file)
    for count, ticker in enumerate(stocks):
        df = deepcopy(data[ticker])

        dates = df.index.tolist()
        vals = df['VWAP'].values
        volume = df['Volume'].values
        res = resMap[ticker + '_' + str(aroon_period) + '_mbv_' + str(max_buy_val) + '_msv_' + str(min_sell_val)]
        indicatorVals = removeNaN(aroonMap[ticker + '_' + str(aroon_period)].values)

        r1 = getIndicatorValues(req1, ticker)
        r2 = getIndicatorValues(req2, ticker)
        r3 = getIndicatorValues(req3, ticker)
        r4 = getIndicatorValues(req4, ticker)

        max_len = getMaxDataPoints(res, indicatorVals, r1, r2, r3, r4)
        dates = getUsableDataPoints(dates, max_len)
        vals = getUsableDataPoints(vals, max_len)
        volume = getUsableDataPoints(volume, max_len)
        res = getUsableDataPoints(res, max_len)
        indicatorVals = getUsableDataPoints(indicatorVals, max_len)
        r1 = getUsableDataPoints(r1, max_len)
        r2 = getUsableDataPoints(r2, max_len)
        r3 = getUsableDataPoints(r3, max_len)
        r4 = getUsableDataPoints(r4, max_len)

        amt = np.zeros(len(res))
        gain = np.zeros(len(res))
        day_diff = np.zeros(len(res)).astype(int)
        sell_days = np.zeros(len(res)).astype(int)
        buy_signal = 0
        sell_signal = 0

        i = 1
        while i < len(res) - 1:
            if res[i] > buy_signal and checkRequirement(req1, i, r1) and checkRequirement(req2, i, r2):
                addTrade(trades, ticker, dates[i + 1][1], vals[i + 1], indicatorVal=indicatorVals[i],
                         volume=volume[i + 1])
                for j in range(i + 1, len(res) - 1):
                    if res[j] < sell_signal and checkRequirement(req3, j, r3) and checkRequirement(req4, j, r4):
                        addTrade(trades, ticker, dates[j + 1][1], vals[j + 1], isBuy=False)
                        amt[j] = vals[j] - vals[i]
                        gain[j] = (vals[j] - vals[i]) * 100 / vals[i]
                        day_diff[j] = j - i
                        i = j
                        break

                    if j == len(res) - 2:
                        i = j

            i += 1

        # to match the df with the values of the arrays
        df.drop(df.head(len(df) - max_len).index, inplace=True)

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

    group = file[32: len(file) - 8]

    result[10] = backTestStrategy(trades, group == '500')
    result[11] = backTestStrategy(trades, group == '500', date='2017-01-01')

    print('NAV', result[10])
    print('NAV 2017', result[11])
    print('\n')

    row = [group, max_buy_val, min_sell_val, aroon_period, req1.__repr__(), req2.__repr__(), req3.__repr__(),
           req4.__repr__()]


    # check for diff append and extend https://stackoverflow.com/a/252711/5512020
    row.extend([element for element in result])
    # without newline = '' , a row is skipped in csv
    with open('optimized_aroon.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    f.close()


def analyseAROONOSC():
    periods = np.array([6, 8, 10, 12, 14, 16])
    maxBuyVals = np.array([-40, -50, -60])
    minSellVals = np.array([40, 50, 60])
    adx_periods = np.array([6, 8, 10, 12, 14, 16])
    adx_mins = np.array([25, 30, 35])

    count = 0
    for i in range(len(periods)):
        req3 = ReqParam()
        req3.condition = 5
        req3.indicatorPeriod = periods[i]
        req3.isBuy = False

        getAROONOSC(periods[i])

        for j in range(len(maxBuyVals)):
            for k in range(len(minSellVals)):
                if (periods[i] == 6 or periods[i] == 8) and minSellVals[k] == 40:
                    break

                getRes(maxBuyVals[j], minSellVals[k], periods[i])

                for l in range(len(adx_periods)):
                    for m in range(len(adx_mins)):
                        req1 = ReqParam()
                        req1.condition = 1
                        req1.indicatorPeriod = adx_periods[l]
                        req1.adx_min = adx_mins[m]
                        getRequirementData(req1)

                        for f in allFiles:
                            count += 1
                            print("Analyses: ", f[32: len(f) - 8], count, periods[i], maxBuyVals[j],
                                  minSellVals[k], adx_periods[l], adx_mins[m])

                            doCalculation(f, max_buy_val=maxBuyVals[j], min_sell_val=minSellVals[k],
                                          aroon_period=periods[i], req1=req1, req3=req3)


pd.options.mode.chained_assignment = None

st = int(round(time.time() * 1000))

path = r'F:\data\nse500\indices'
allFiles = glob.glob(path + "\*.csv")

getData()
analyseAROONOSC()

print(int(round(time.time() * 1000)) - st)