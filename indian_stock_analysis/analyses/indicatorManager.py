import talib
import pandas as pd
import numpy as np
import csv
import math
import time

# Check the answer https://stackoverflow.com/a/20627316/5512020
# to prevent the warning: A value is trying to be set on a copy of a slice from a DataFrame.
pd.options.mode.chained_assignment = None  # default='warn'

total_res = np.zeros(15)

data = pd.read_csv('../ind_nifty500list.csv')
nifty500 = data['Symbol'].values
for i in range(len(nifty500)):
    if nifty500[i] == 'FAGBEARING':
        nifty500[i] = 'SCHAEFFLER'


def analysePrevDays(prevCloses):
    plusCount = 0
    for i in range(1, len(prevCloses)):
        if prevCloses[i] > prevCloses[i - 1]:
            plusCount += 1

    return plusCount

def analyseStrategy(res):
    totalValue = 0
    effValue = 0

    if not np.isnan(res[9]) or res[9] == 0:
        totalValue += 0
    elif 150 < res[9] < 600:
        totalValue += 1
    elif res[9] < 2000:
        totalValue += 2
    elif res[9] < 5000:
        totalValue += 3
    elif res[9] < 10000:
        totalValue += 4
    elif res[9] < 15000:
        totalValue += 3
    elif res[9] < 20000:
        totalValue += 2
    elif not np.isnan(res[9]):
        totalValue += 1


    if np.isnan(res[3]) or res[3] < 1:
        effValue += 0
    else:
        effValue += math.floor(res[3] - 1.5)
    totalValue += effValue

    eff = res[1] / res[9] if not np.isnan(res[1]) and not np.isnan(res[9]) and res[9] != 0 else 0
    if eff < 0.6:
        effValue += 0
    elif eff < 0.67:
        effValue += 1
    elif eff < 0.75:
        effValue += 2
    elif eff < 0.8:
        effValue += 3
    elif eff < 0.85:
        effValue += 4
    elif eff < 0.9:
        effValue += 5
    elif eff < 0.93:
        effValue += 6
    elif eff < 0.96:
        effValue += 7
    else:
        effValue += 8

    totalValue += effValue

    if np.isnan(res[6]) or res[6] > 120:
        totalValue += 0
    elif res[6] > 90:
        totalValue += 1
    elif res[6] > 65:
        totalValue += 2
    elif res[6] > 45:
        totalValue += 3
    elif res[6] > 30:
        totalValue += 4
    elif res[6] > 20:
        totalValue += 5
    elif res[6] > 13:
        totalValue += 6
    else:
        totalValue += 7

    return effValue, totalValue


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

def getIndicatorValues(indicator, needs, df, period, fastperiod=0, slowperiod=0, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    if needs == 0:
        return getattr(talib, indicator)(df['Close'].values, timeperiod=period)
    elif needs == 1:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, timeperiod=period)
    elif needs == 2:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values,
                                                  timeperiod=period)
    elif needs == 3:
        return getattr(talib, indicator)(df['Open'].values, df['High'].values, df['Low'].values,
                                                  df['Close'].values)
    elif needs == 4:
        return getattr(talib, indicator)(df['Close'].values, fastperiod=fastperiod,
                                                  slowperiod=slowperiod, matype=0)
    elif needs == 5:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values,
                                                  df['Volume'].values, timeperiod=period)
    elif needs == 6:
        return getattr(talib, indicator)(df['Open'].values, df['High'].values, df['Low'].values,
                                                  df['Close'].values, timeperiod1=timeperiod1,
                                                  timeperiod2=timeperiod2, timeperiod3=timeperiod3)
    elif needs == 7:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values,
                                                  df['Volume'].values, fastperiod=fastperiod, slowperiod=slowperiod)
    elif needs == 8:
        return getattr(talib, indicator)(df['Close'].values, df['Volume'].values)
    elif needs == 9:
        return getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values)


def analyze_indicator(indicator, max_buy_val, min_sell_val, period, fastperiod=0, slowperiod=0, timeperiod1=7,
                      timeperiod2=14, timeperiod3=28):

    needs = getColumnKey(indicator)

    # without count the ticker name is incorrect
    for count, ticker in enumerate(nifty500):
        total_df = pd.read_csv("F:/data/nse500/selected/{}.csv".format(ticker))
        total_df.reset_index(inplace=True)
        df = total_df[['Symbol', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
        df.set_index(['Symbol', 'Date'], inplace=True)

        df[indicator] = getIndicatorValues(indicator, needs, df, period, fastperiod=fastperiod, slowperiod=slowperiod, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
        df.dropna(axis=0, inplace=True)

        df['Res'] = np.where(df[indicator] < max_buy_val, 1, np.where(df[indicator] > min_sell_val, -1, 0))
        df['maV'] = df['Volume'].rolling(100).mean()
        df['anaV'] = np.where(df['Volume'] > 1.25 * df['maV'], 1, 0)
        df['Count'] = np.arange(len(df))[:]

        try:
            _, __, df['MACDHist'] = talib.MACD(df['Close'].values)
        except:
            print(ticker)

        df.dropna(axis=0, inplace=True)

        op = df['Res'].values
        vals = df['Close'].values
        amt = np.zeros(len(df['Res']))
        gain = np.zeros(len(df['Res']))
        day_diff = np.zeros(len(df['Res'])).astype(int)
        sell_days = np.zeros(len(df['Res'])).astype(int)
        anaV = df['anaV'].values
        hist = df['MACDHist'].values

        buy_signal = 0
        sell_signal = 0
        closePosition = -1

        anaDF = df[['Count', 'Res', 'Close']][((df['Res'] > buy_signal) & (df['MACDHist'] < 0) & (df['anaV'] == 1)) |
                      ((df['Res'] < sell_signal) & (df['MACDHist'] > 0) & (df['anaV'] == 1))]

        if ticker == 'RELIANCE':
            print(anaDF)

        for i in range(len(op)):
            if op[i] > buy_signal and hist[i] < 0 and anaV[i] == 1 and i > closePosition:
                for j in range(i + 1, len(op)):
                    if op[j] < sell_signal and hist[j] > 0 and anaV[j] == 1:
                        amt[j] = vals[j] - vals[i]
                        gain[j] = (vals[j] - vals[i]) * 100 / vals[i]
                        day_diff[j] = j - i
                        closePosition = j
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

        if ticker == 'RELIANCE':
            print(df[indicator])
            print(df['anaV'], df['MACDHist'], df['Res'])
            print(len(df[(df['MACDHist'] < 0) & (df['anaV'] == 1) & (df['Res'] > 0)]))
            print(len(df['Res'][df['Res'] > 0]), len(df['Res'][df['Res'] < 0]))
            print('Total days', len(df['Gain']))
            print('+', len(df['Gain'][df['Gain'] > 0]))
            print('-', len(df['Gain'][df['Gain'] < 0]))
            print('avg gain', df['Gain'][(df['Gain'] > 0) | (df['Gain'] < 0)].mean())
            print('avg + gain', df['Gain'][df['Gain'] > 0].mean())
            print('avg - gain', df['Gain'][df['Gain'] < 0].mean())
            print('avg day diff', df['Day_Diff'][(df['Gain'] > 0) | (df['Gain'] < 0)].mean())
            print('avg + day diff', df['Day_Diff'][df['Gain'] > 0].mean())
            print('avg - day diff', df['Day_Diff'][df['Gain'] < 0].mean())
            print('Total buys', len(df['Gain'][df['Gain'] != 0]))
            print('Total sells', len(df['Sell_Days'][df['Sell_Days'] > 0]))

    total_res[3] = total_res[3] / total_res[9]
    total_res[4] = total_res[4] / total_res[1]
    total_res[5] = total_res[5] / total_res[2]
    total_res[6] = total_res[6] / total_res[9]
    total_res[7] = total_res[7] / total_res[1]
    total_res[8] = total_res[8] / total_res[2]
    total_res[13], total_res[14] = analyseStrategy(total_res)

    print('Total days', total_res[0])
    print('Total buys', total_res[9])
    print('Total sell days', total_res[10])
    print('+', total_res[1])
    print('-', total_res[2])
    print('avg gain', total_res[3])
    print('avg + gain', total_res[4])
    print('avg - gain', total_res[5])
    print('avg day diff', total_res[6])
    print('avg + day diff', total_res[7])
    print('avg - day diff', total_res[8])
    print('stop loss positive count', total_res[11])
    print('stop loss negative count', total_res[12])
    print('effective value', total_res[13])
    print('total value', total_res[14])

    row = [indicator, max_buy_val, min_sell_val, period]

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
    row.extend([element for element in total_res])
    csvRow = ','.join(map(str, row))
    with open('analysis.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csvRow)
    f.close()


def analyseRSI():
    maxBuyVals = np.array([-50])
    minSellVals = np.array([50])#,65,70,75])
    periods = np.array([10])#,12,14,16,18,20])
    for i in range(len(maxBuyVals)):
        for j in range(len(minSellVals)):
            for k in range(len(periods)):
                analyze_indicator(indicator='CMO', max_buy_val=maxBuyVals[i], min_sell_val=minSellVals[j], period=periods[k])



st = time.time()
analyseRSI()
print('Total time', time.time() - st)