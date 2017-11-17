from copy import deepcopy
import csv
from datetime import datetime
import glob
import operator
from data import getStocks, getData
from objects import Indicator, ReqParam, Stock, DayTrades
from talib_processes import getIndicatorValues, getPeriodStr, getRequirementData
import pandas as pd
import numpy as np
import util
import time
from util import removeNaN


indicatorMap = {}
resMap = {}
adxMap = {}
chaikinMap = {}
histMap = {}
volMap = {}
natrMap = {}


def getRes(indicator: Indicator, max_buy_val, min_sell_val):
    stocks = getStocks()
    for count, ticker in enumerate(stocks):
        df = pd.DataFrame()
        df[indicator.name] = indicatorMap.get(ticker + getPeriodStr(indicator))
        df['Res'] = np.where(df[indicator.name] <= max_buy_val, 1, np.where(df[indicator.name] >= min_sell_val, -1, 0))
        resMap[ticker + getPeriodStr(indicator) + '_mbv_' + str(max_buy_val) + '_msv_' + str(min_sell_val)] = util.removeNaN(df['Res'].values)


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
    elif req.condition == 6:
        if position is None or vals is None or req.natr_min is None:
            raise ValueError('Need values for position, adx and adx_min')
        return vals[position] > req.natr_min
    else:
        raise ValueError('Need condition needs to be between 0 & 6')


def getRequirementValues(req: ReqParam, ticker, indicator: Indicator):
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
        return indicatorMap[ticker + getPeriodStr(indicator)].values
    elif req.condition == 6:
        return natrMap[ticker + '_' + str(req.indicatorPeriod)]


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


def doCalculation(file, max_buy_val, min_sell_val, indicator, req1=ReqParam(), req2=ReqParam(), req3=ReqParam(), req4=ReqParam()):
    trades = {}
    total_res = np.zeros(15)
    stocks = getStocks(file)
    for count, ticker in enumerate(stocks):
        df = deepcopy(data[ticker])

        dates = df.index.tolist()
        vals = df['VWAP'].values
        volume = df['Volume'].values
        res = resMap[ticker + getPeriodStr(indicator) + '_mbv_' + str(max_buy_val) + '_msv_' + str(min_sell_val)]
        indicatorVals = removeNaN(indicatorMap[ticker + getPeriodStr(indicator)].values)

        r1 = getRequirementValues(req1, ticker, indicator)
        r2 = getRequirementValues(req2, ticker, indicator)
        r3 = getRequirementValues(req3, ticker, indicator)
        r4 = getRequirementValues(req4, ticker, indicator)

        max_len = util.getMaxDataPoints(res, indicatorVals, r1, r2, r3, r4)
        dates = util.getUsableDataPoints(dates, max_len)
        vals = util.getUsableDataPoints(vals, max_len)
        volume = util.getUsableDataPoints(volume, max_len)
        res = util.getUsableDataPoints(res, max_len)
        indicatorVals = util.getUsableDataPoints(indicatorVals, max_len)
        r1 = util.getUsableDataPoints(r1, max_len)
        r2 = util.getUsableDataPoints(r2, max_len)
        r3 = util.getUsableDataPoints(r3, max_len)
        r4 = util.getUsableDataPoints(r4, max_len)

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

    result = np.zeros(13)
    result[0] = total_res[0]
    result[1] = total_res[1]
    result[2] = total_res[2]
    result[3] = total_res[1]/total_res[9] if total_res[1] > 100 else 0
    result[4] = total_res[3] / total_res[9]
    result[5] = total_res[4] / total_res[1]
    result[6] = total_res[5] / total_res[2]
    result[7] = total_res[6] / total_res[9]
    result[8] = total_res[7] / total_res[1]
    result[9] = total_res[8] / total_res[2]
    result[10] = total_res[9]

    group = file[32: len(file) - 8]

    result[11] = backTestStrategy(trades, group == '500')
    result[12] = backTestStrategy(trades, group == '500', date='2017-01-01')

    row = [indicator.name, group, max_buy_val, min_sell_val, indicator.period, indicator.fastperiod, indicator.slowperiod, indicator.timeperiod1, indicator.timeperiod2, indicator.timeperiod3, req1.__repr__(), req2.__repr__(), req3.__repr__(),
           req4.__repr__()]

    # check for diff append and extend https://stackoverflow.com/a/252711/5512020
    row.extend([element for element in result])
    # without newline = '' , a row is skipped in csv
    with open('./analyses/analyses.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    f.close()


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
                money += buyDict.get(stock.symbol)[0] * stock.wap * 0.997
                nav += buyDict.get(stock.symbol)[0] * (stock.wap * 0.997 - buyDict.get(stock.symbol)[1])
                buyDict[stock.symbol] = (0, 0)

        # // gives int division so no decimal
        max_stocks = 15 if isNifty500 else 4
        max_per_stock = nav // max_stocks
        sortedBuys = sorted(dayTrade.buys, key=operator.attrgetter('indicatorVal'))
        for s in sortedBuys:
            amount = min(max_per_stock, money)
            quantity = min(amount // (s.wap * 1.003), s.max_quantity)
            buyDict[s.symbol] = (quantity, s.wap)
            money -= amount - amount % s.wap

    return nav - iniVal


def analyseIndicator(count):
    periods = np.array([6, 8, 10, 12, 14, 16, 18])
    maxBuyVals = np.array([-85, -80, -75, -70])
    minSellVals = np.array([-25, -20, -15])
    adx_periods = np.array([6, 8, 10, 12, 14])
    adx_mins = np.array([25, 30, 35])

    for i in range(len(periods)):
        indicator = Indicator()
        indicator.period = periods[i]
        indicator.name = 'WILLR'

        req2 = ReqParam()
        req2.condition = 5
        req2.indicatorPeriod = periods[i]
        req2.isBuy = True

        req3 = ReqParam()
        req3.condition = 5
        req3.indicatorPeriod = periods[i]
        req3.isBuy = False

        getIndicatorValues(indicator, indicatorMap)

        for j in range(len(maxBuyVals)):
            for k in range(len(minSellVals)):
                getRes(indicator, maxBuyVals[j], minSellVals[k])

                for l in range(len(adx_periods)):
                    for m in range(len(adx_mins)):
                        req1 = ReqParam()
                        req1.condition = 1
                        req1.indicatorPeriod = adx_periods[l]
                        req1.adx_min = adx_mins[m]
                        getRequirementData(req1, data, adxMap)

                        for f in allFiles:
                            count += 1
                            print("Analyses: ", f[32: len(f) - 8], count, periods[i], maxBuyVals[j],
                                  minSellVals[k], adx_periods[l], adx_mins[m])

                            doCalculation(f, max_buy_val=maxBuyVals[j], min_sell_val=minSellVals[k], indicator=indicator, req1=req1, req2=req2, req3=req3)


pd.options.mode.chained_assignment = None

st = int(round(time.time() * 1000))
count = 0

path = r'F:\data\nse500\indices'
allFiles = glob.glob(path + "\*.csv")

data = getData()
analyseIndicator(count)
totalTime = int(round(time.time() * 1000)) - st
print('Total Time:', totalTime)