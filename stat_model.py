import pickle
import talib
import pandas as pd

import numpy as np

res = np.zeros(15).astype(int)
total_res = np.zeros(10)

def compile_data():
    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    data = {}

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

            df.drop(["Adj Close"], axis=1, inplace=True)

            df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cADX'] = np.where(df.ADX > 65, 2, np.where(df.ADX > 25, 1, 0))

            df['RSI'] = talib.RSI(df['Close'].values, 14)
            df['cRSI'] = np.where(df.RSI > 65, -1, np.where(df.RSI < 30, 1, 0))

            # CCI(high, low, close, timeperiod=14)
            df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cCCI'] = np.where(df.CCI > 100, -1, np.where(df.CCI < 0, 1, 0))

            # WILLR(high, low, close, timeperiod=14)
            df['WILLR'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, 14)
            df['cWILLR'] = np.where(df.WILLR < -80, 1, np.where(df.WILLR > -20, -1, 0))

            # TRIX(close, timeperiod=30)
            # df['TRIX'] = talib.TRIX(df['Close'].values, 30)
            # df['cTRIX'] = np.where(df['TRIX'].shift(1) > df['TRIX'] and df['TRIX'].shift(-1) > df['TRIX'] and  df['TRIX'].shift(1) < 0 and df['TRIX'].shift(-1) < 0, 2,
            #                        np.where(df['TRIX'].shift(1) < df['TRIX'] and df['TRIX'].shift(-1) < df['TRIX'] and df['TRIX'].shift(1) > 0 and df['TRIX'].shift(-1) > 0, -2, 0))

            # CMO(close, timeperiod=14)
            df['CMO'] = talib.CMO(df['Close'].values, 14)
            df['cCMO'] = np.where(df.CMO > 50, -1, np.where(df.CMO < -50, 1, 0))

            # AROONOSC(high, low, timeperiod=14)
            df['AROONOSC'] = talib.AROONOSC(df['High'].values, df['Low'].values, 14)
            df['cAROONOSC'] = np.where(df.AROONOSC > 50, -1, np.where(df.AROONOSC < -50, 1, 0))

            df.dropna(inplace=True)

            df['temp'] = df['cRSI'] + df['cCCI'] + df['cWILLR'] + df['cCMO'] + df['cAROONOSC']
            df['Res'] = np.where(df['temp'] > 0, df['temp'] + df['cADX'], np.where(df['temp'] < 0, df['temp'] - df['cADX'], 0))

            df.drop(["Open", "High", "Low", "Volume"], axis=1, inplace=True)
            data[ticker] = df

            for i in range(-6, 7):
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

        except Exception as e:
            print(e)

    panel = pd.Panel(data)


compile_data()

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