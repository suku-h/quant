import pickle
import talib
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import copy

import numpy as np

res = np.zeros(15).astype(int)
total_res = np.zeros(12)

pos = []
neg = []

def assessCurve(pd_curve, window, isIndicator, **kwargs):
    sell_condition = kwargs.get('sell_condition', None)
    buy_condition = kwargs.get('buy_condition', None)

    # print(sell_condition, buy_condition)

    df = pd.DataFrame(0, pd_curve.index, columns=['curve_crawl'])
    df['curve'] = pd.Series.copy(pd_curve)

    c = df['curve'].values
    cc = df['curve_crawl'].values

    for i in range(1, len(c) - 1):
        if (c[i-1] > c[i] > c[i+1]) or (c[i-1] < c[i] < c[i+1]):
            if cc[i-1] == 100:
                cc[i] = 1
            else:
                cc[i] = cc[i-1] + 1
        elif c[i] > c[i-1] and c[i] > c[i+1]:
            cc[i] = 100
        else:
            cc[i] = 0


    #print(df)

    ass = np.zeros(len(c))

    # for i in cc makes it run till the highest value in cc
    if not isIndicator:
        for i in range(1, len(c) - window - 14):
            if cc[i] == 100:
                # print('d', i,cc[i-20:i+20])
                for j in range(i + 8, i + window):
                    if cc[j] == 100:  # and np.max(cc[j+1:j+10]) != 100:
                        if c[i] < c[j] and c[j] > 1.1 * c[i]:
                            ass[j + 4: j + 14] = -1
                        elif c[j] < c[i] and c[i] > 1.1 * c[j]:
                            ass[j + 4: j + 14] = 1
    else:
        for i in range(1, len(c) - window - 14):
            if cc[i] == 100:
                # print('d', i,cc[i-20:i+20])
                for j in range(i + 8, i + window):
                    if cc[j] == 100:  # and np.max(cc[j+1:j+10]) != 100:
                        if c[i] > c[j] > sell_condition and c[i] > 1.1 * c[j]:
                            ass[j + 4: j + 14] = -1
                        elif buy_condition < c[i] < c[j] and c[j] > 1.1 * c[i]:
                            ass[j + 4: j + 14] = 1

    return ass[:]



def compile_data():
    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    # without count the ticker name is incorrect
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

            window = 40
            df.drop(["Adj Close"], axis=1, inplace=True)

            curve = savgol_filter(df['Close'].values, 21, 5)
            df['cClose'] = curve[:]
            df['acClose'] = assessCurve(df['cClose'], window=window, isIndicator=False)

            df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cADX'] = np.where(df.ADX > 65, 2, np.where(df.ADX > 30, 1, 0))

            df['RSI'] = talib.RSI(df['Close'].values, 14)
            df['cRSI'] = assessCurve(df['RSI'], window=window, buy_condition=40, sell_condition=65, isIndicator=True)# np.where(df.RSI > 65, -1, np.where(df.RSI < 30, 1, 0))

            # CCI(high, low, close, timeperiod=14)
            df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cCCI'] = assessCurve(df['CCI'], window=window, buy_condition=0, sell_condition=100, isIndicator=True)

            # WILLR(high, low, close, timeperiod=14)
            df['WILLR'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cWILLR'] = assessCurve(df['WILLR'], window=window, buy_condition = -80, sell_condition = -20, isIndicator=True)# np.where(df.WILLR < -80, 1, np.where(df.WILLR > -20, -1, 0))

            # TRIX(close, timeperiod=30)
            # df['TRIX'] = talib.TRIX(df['Close'].values, 30)
            # df['cTRIX'] = np.where(df['TRIX'].shift(1) > df['TRIX'] and df['TRIX'].shift(-1) > df['TRIX'] and  df['TRIX'].shift(1) < 0 and df['TRIX'].shift(-1) < 0, 2,
            #                        np.where(df['TRIX'].shift(1) < df['TRIX'] and df['TRIX'].shift(-1) < df['TRIX'] and df['TRIX'].shift(1) > 0 and df['TRIX'].shift(-1) > 0, -2, 0))


            # CMO(close, timeperiod=14)
            df['CMO'] = talib.CMO(df['Close'].values, 14)
            df['cCMO'] = assessCurve(df['CMO'], window=window, buy_condition=-50, sell_condition=50, isIndicator=True)# np.where(df.CMO > 50, -1, np.where(df.CMO < -50, 1, 0))

            # AROONOSC(high, low, timeperiod=14)
            df['AROONOSC'] = talib.AROONOSC(df['High'].values, df['Low'].values, 14)
            df['cAROONOSC'] = assessCurve(df['AROONOSC'], window=window, buy_condition=-50, sell_condition=50, isIndicator=True)#np.where(df.AROONOSC > 50, -1, np.where(df.AROONOSC < -50, 1, 0))

            df.dropna(inplace=True)

            df['temp'] = df['cRSI'] + df['cCCI'] + df['cWILLR'] + df['cCMO'] + df['cAROONOSC']
            df['Res'] = np.where(df['temp'] > 0, df['temp'] + df['cADX'], np.where(df['temp'] < 0, df['temp'] - df['cADX'], 0))

            df.drop(["Open", "High", "Low", "Volume"], axis=1, inplace=True)

            for i in range(-7, 8):
                res[i + 7] += len(df['Res'][df['Res'] == i])

            op = df['Res'].values
            vals = df['Close'].values
            amt = np.zeros(len(df['Res']))
            gain = np.zeros(len(df['Res']))
            day_diff = np.zeros(len(df['Res'])).astype(int)

            buy_signal = 3
            sell_signal = -3
            for i in range(len(op)):
                if op[i] > buy_signal:
                    for j in range(i+1, len(op)):
                        if op[j] < sell_signal:
                            amt[i] = vals[j] - vals[i]
                            gain[i] = (vals[j] - vals[i]) * 100 / vals[i]
                            day_diff[i] = j - i
                            break



            df['Amt'] = amt[:]
            df['Gain'] = gain[:]
            df['Day_Diff'] = day_diff[:]

            df.drop(["temp"], axis=1, inplace=True)

            total_res[0] += len(df['Gain'])
            total_res[1] += len(df['Gain'][df['Gain'] > 0])
            total_res[2] += len(df['Gain'][df['Gain'] < 0])
            total_res[3] += df['Gain'][df['Gain'] != 0].sum()
            total_res[4] += df['Gain'][df['Gain'] > 0].sum()
            total_res[5] += df['Gain'][df['Gain'] < 0].sum()
            total_res[6] += df['Day_Diff'][df['Gain'] != 0].sum()
            total_res[7] += df['Day_Diff'][df['Gain'] > 0].sum()
            total_res[8] += df['Day_Diff'][df['Gain'] < 0].sum()
            total_res[9] += len(df['Res'][df['Res'] > buy_signal])
            total_res[10] += len(df['acClose'][df['acClose'] == 1])//10
            total_res[11] += len(df['acClose'][df['acClose'] == -1])//10


            if ticker == 'AAP':
                print(df.tail(80))
                print('Total days', len(df['Gain']))
                print('+', len(df['Gain'][df['Gain'] > 0]))
                print('-', len(df['Gain'][df['Gain'] < 0]))
                print('avg gain', df['Gain'][df['Gain'] != 0].mean())
                print('avg + gain', df['Gain'][df['Gain'] > 0].mean())
                print('avg - gain', df['Gain'][df['Gain'] < 0].mean())
                print('avg day diff', df['Day_Diff'][df['Day_Diff'] != 0].mean())
                print('avg + day diff', df['Day_Diff'][df['Gain'] > 0].mean())
                print('avg - day diff', df['Day_Diff'][df['Gain'] < 0].mean())
                print('Total buys', len(df['Gain'][df['Gain'] != 0]))
                print('ass +', len(df['acClose'][df['acClose'] == 1])//10)
                print('ass -', len(df['acClose'][df['acClose'] == -1])//10)

                x = range(len(df['Close']))
                close = df['Close'].values
                # plt.plot(x, close/10)
                # plt.plot(x, curve/10, color='red')
                # plt.plot(x, df['acClose'].values, color='green')
                # plt.show()

            df['ticker'] = ticker
            df.reset_index(inplace=True)
            df.set_index(['ticker', 'Date'], inplace=True)

            pos.append(df[df['Gain'] > 0])
            neg.append(df[df['Gain'] < 0])

        except Exception as e:
            print(e)

compile_data()
#
# pos = pd.concat(pos)
# neg = pd.concat(neg)
# pos.to_csv('pos.csv')
# neg.to_csv('neg.csv')

print(res)
print('Total days', total_res[0])
print('Total buys', total_res[9])
print('+', total_res[1])
print('-', total_res[2])
print('avg gain', total_res[3]/total_res[9])
print('avg + gain', total_res[4]/total_res[1])
print('avg - gain', total_res[5]/total_res[2])
print('avg day diff', total_res[6]/total_res[9])
print('avg + day diff', total_res[7]/total_res[1])
print('avg - day diff', total_res[8]/total_res[2])
print('ass +', total_res[10])
print('ass -', total_res[11])