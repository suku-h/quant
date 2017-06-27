import pickle
import talib
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
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
        if (c[i - 1] > c[i] > c[i + 1]) or (c[i - 1] < c[i] < c[i + 1]):
            if cc[i - 1] == 100:
                cc[i] = 1
            else:
                cc[i] = cc[i - 1] + 1
        elif c[i] > c[i - 1] and c[i] > c[i + 1]:
            cc[i] = 100
        else:
            cc[i] = 0


    ass = np.zeros(len(c))
    s = np.zeros(len(c))

    # np.where returns a tuple (a,b) where b is the position
    # so (a,b)[0][:] returns a numpy array of only positions
    # this should finish calculating significantly faster than the previous nested loops
    cc100 = np.where(cc == 100)[0][:]
    for i in range(len(cc100) - 1):
        j = i + 1
        while j < len(cc100) and cc100[j] < cc100[i] + 8:
            j += 1

        while j < len(cc100) and cc100[j] < cc100[i] + window:
            k = 14
            if not isIndicator and c[cc100[i]] * 1.2 < c[cc100[j]]:
                s[cc100[i]] = -1
                s[cc100[j]] = -2
                ass[cc100[j] + 4: cc100[j] + 14] = -1
            elif isIndicator and sell_condition < c[cc100[j]] * 1.2 < c[cc100[i]]:
                s[cc100[i]] = -1
                s[cc100[j]] = -2
                ass[cc100[j] + 4: cc100[j] + 14] = -1
            elif not isIndicator and c[cc100[j]] * 1.2 < c[cc100[i]]:
                s[cc100[i]] = 1
                s[cc100[j]] = 2
                ass[cc100[j] + 4: cc100[j] + 14] = 1
            elif isIndicator and c[cc100[i]] < c[cc100[j]] * 1.2 < buy_condition:
                s[cc100[i]] = 1
                s[cc100[j]] = 2
                ass[cc100[j] + 4: cc100[j] + 14] = 1
            else:
                k = 1
            j += k

    return ass[:], s[:]


def compile_data():
    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    # without count the ticker name is incorrect
    window = 50

    for count, ticker in enumerate(tickers):
        if ticker == 'AAP':
        # try:
            filepath = "F:/data/sp500/{}.csv".format(ticker)
            df = pd.read_csv(filepath)
            df.set_index('Date', inplace=True)
            ratio = df["Close"] / df["Adj Close"]
            df["Open"] = df["Open"] / ratio
            df["High"] = df["High"] / ratio
            df["Low"] = df["Low"] / ratio
            df["Volume"] = df["Volume"] / ratio
            df["Close"] = df["Adj Close"]

            df.drop(["Adj Close"], axis=1, inplace=True)

            curve = savgol_filter(df['Close'].values, 21, 5)
            df['cClose'] = curve[:]
            df['acClose'], df['sClose'] = assessCurve(df['cClose'], window=window, isIndicator=False)

            df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cADX'] = np.where(df.ADX > 65, 2, np.where(df.ADX > 30, 1, 0))

            df['RSI'] = talib.RSI(df['Close'].values, 14)
            df['cRSI'], df['sRSI'] = assessCurve(df['RSI'], window=window, buy_condition=40, sell_condition=65,
                                     isIndicator=True)  # np.where(df.RSI > 65, -1, np.where(df.RSI < 30, 1, 0))

            # CCI(high, low, close, timeperiod=14)
            df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cCCI'], df['sCCI'] = assessCurve(df['CCI'], window=window, buy_condition=0, sell_condition=100, isIndicator=True)

            # WILLR(high, low, close, timeperiod=14)
            df['WILLR'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cWILLR'], df['sWILLR'] = assessCurve(df['WILLR'], window=window, buy_condition=-80, sell_condition=-20,
                                       isIndicator=True)  # np.where(df.WILLR < -80, 1, np.where(df.WILLR > -20, -1, 0))

            # TRIX(close, timeperiod=30)
            # df['TRIX'] = talib.TRIX(df['Close'].values, 30)
            # df['cTRIX'] = np.where(df['TRIX'].shift(1) > df['TRIX'] and df['TRIX'].shift(-1) > df['TRIX'] and  df['TRIX'].shift(1) < 0 and df['TRIX'].shift(-1) < 0, 2,
            # np.where(df['TRIX'].shift(1) < df['TRIX'] and df['TRIX'].shift(-1) < df['TRIX'] and df['TRIX'].shift(1) > 0 and df['TRIX'].shift(-1) > 0, -2, 0))


            # CMO(close, timeperiod=14)
            df['CMO'] = talib.CMO(df['Close'].values, 14)
            df['cCMO'], df['sCMO'] = assessCurve(df['CMO'], window=window, buy_condition=-50, sell_condition=50,
                                     isIndicator=True)  # np.where(df.CMO > 50, -1, np.where(df.CMO < -50, 1, 0))

            # AROONOSC(high, low, timeperiod=14)
            df['AROONOSC'] = talib.AROONOSC(df['High'].values, df['Low'].values, 14)
            df['cAROONOSC'], df['sAROONOSC'] = assessCurve(df['AROONOSC'], window=window, buy_condition=-50, sell_condition=50,
                                          isIndicator=True)  #np.where(df.AROONOSC > 50, -1, np.where(df.AROONOSC < -50, 1, 0))

            df.dropna(inplace=True)

            df['temp'] = df['cRSI'] + df['cCCI'] + df['cWILLR'] + df['cCMO'] + df['cAROONOSC']
            df['Res'] = np.where(df['temp'] > 0, df['temp'] + df['cADX'],
                                 np.where(df['temp'] < 0, df['temp'] - df['cADX'], 0))


            for i in range(-7, 8):
                res[i + 7] += len(df['Res'][df['Res'] == i])

            op = df['Res'].values
            close_op = df['acClose'].values
            vals = df['Close'].values
            amt = np.zeros(len(df['Res']))
            gain = np.zeros(len(df['Res']))
            day_diff = np.zeros(len(df['Res'])).astype(int)

            buy_signal = 2
            sell_signal = -2
            for i in range(len(op)):
                if op[i] > buy_signal and close_op[i] == 1:
                    for j in range(i + 1, len(op)):
                        if op[j] < sell_signal and close_op[j] == -1:
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
            total_res[10] += len(df['acClose'][df['acClose'] == 1]) // 10
            total_res[11] += len(df['acClose'][df['acClose'] == -1]) // 10

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
                print('ass +', len(df['acClose'][df['acClose'] == 1]) // 10)
                print('ass -', len(df['acClose'][df['acClose'] == -1]) // 10)

                fig = plt.figure()
                plt.title('AAP')
                print(df['sClose'])
                print(len(df['sClose'][df['sClose'] == -1]),len(df['sClose'][df['sClose'] == 1]))
                date = df.index.values
                ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=4, colspan=1)
                y = df['Close'].values
                # plot does not work with dates hence use plot_date
                ax1.plot_date(date, y, '-', linewidth=1, color='m')
                plt.ylabel('Price')
                n = df['sClose'].values
                for i in range(len(n)):
                    if n[i] == 1:
                        for j in range(window):
                            if n[i+j] == 2:
                                ax1.plot_date([date[i], date[i+j]], [y[i], y[i+j]], '-', lw=1, color='r')
                                break
                    elif n[i] == -1:
                        for j in range(window):
                            if n[i + j] == -2:
                                ax1.plot([date[i], date[i + j]], [y[i], y[i + j]], '-', lw=1, color='g')
                                break

                ax2 = plt.subplot2grid((8, 1), (4, 0), rowspan=2, colspan=1, sharex=ax1)
                ax2.plot_date(date, df['RSI'].values, '-', linewidth= 1, color='gray')
                plt.ylabel('RSI')
                ax3 = plt.subplot2grid((8, 1), (6, 0), rowspan=2, colspan=1, sharex=ax1)
                ax3.plot_date(date, df['CCI'].values, '-', linewidth=1)
                plt.ylabel('CCI')
                plt.xlabel('Date')

                for label in ax3.xaxis.get_ticklabels():
                    label.set_rotation(45)


                # plt.plot(x, close/10)
                # plt.plot(x, curve/10, color='red')
                # plt.plot(x, df['acClose'].values, color='green')
                plt.show()

            df['ticker'] = ticker
            df.reset_index(inplace=True)
            df.set_index(['ticker', 'Date'], inplace=True)

            pos.append(df[df['Gain'] > 0])
            neg.append(df[df['Gain'] < 0])

        # except Exception as e:
        #     print(e)


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
print('avg gain', total_res[3] / total_res[9])
print('avg + gain', total_res[4] / total_res[1])
print('avg - gain', total_res[5] / total_res[2])
print('avg day diff', total_res[6] / total_res[9])
print('avg + day diff', total_res[7] / total_res[1])
print('avg - day diff', total_res[8] / total_res[2])
print('ass +', total_res[10])
print('ass -', total_res[11])