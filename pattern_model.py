import pickle
import talib
import pandas as pd
import numpy as np

res = np.zeros(15).astype(int)
total_res = np.zeros(10)

pos = []
neg = []

def compile_data():
    with open("sp500tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    for count, ticker in enumerate(tickers):
        # if ticker == 'AAP':
        try:
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

            op = np.zeros(len(df["Open"]))

            fs = [talib.CDL2CROWS,
                  talib.CDL3BLACKCROWS,
                  talib.CDL3INSIDE,
                  talib.CDL3LINESTRIKE,
                  talib.CDL3OUTSIDE,
                  talib.CDL3STARSINSOUTH,
                  talib.CDL3WHITESOLDIERS,
                  talib.CDLABANDONEDBABY,
                  talib.CDLADVANCEBLOCK,
                  talib.CDLBELTHOLD,
                  talib.CDLBREAKAWAY,
                  talib.CDLCLOSINGMARUBOZU,
                  talib.CDLCONCEALBABYSWALL,
                  talib.CDLCOUNTERATTACK,
                  talib.CDLDARKCLOUDCOVER,
                  talib.CDLDOJI,
                  talib.CDLDOJISTAR,
                  talib.CDLDRAGONFLYDOJI,
                  talib.CDLENGULFING,
                  talib.CDLEVENINGDOJISTAR,
                  talib.CDLEVENINGSTAR,
                  talib.CDLGAPSIDESIDEWHITE,
                  talib.CDLGRAVESTONEDOJI,
                  talib.CDLHAMMER,
                  talib.CDLHANGINGMAN,
                  talib.CDLHARAMI,
                  talib.CDLHARAMICROSS,
                  talib.CDLHIGHWAVE,
                  talib.CDLHIKKAKE,
                  talib.CDLHIKKAKEMOD,
                  talib.CDLHOMINGPIGEON,
                  talib.CDLIDENTICAL3CROWS,
                  talib.CDLINNECK,
                  talib.CDLINVERTEDHAMMER,
                  talib.CDLKICKING,
                  talib.CDLKICKINGBYLENGTH,
                  talib.CDLLADDERBOTTOM,
                  talib.CDLLONGLEGGEDDOJI,
                  talib.CDLLONGLINE,
                  talib.CDLMARUBOZU,
                  talib.CDLMATCHINGLOW,
                  talib.CDLMATHOLD,
                  talib.CDLMORNINGDOJISTAR,
                  talib.CDLMORNINGSTAR,
                  talib.CDLONNECK,
                  talib.CDLPIERCING,
                  talib.CDLRICKSHAWMAN,
                  talib.CDLRISEFALL3METHODS,
                  talib.CDLSEPARATINGLINES,
                  talib.CDLSHOOTINGSTAR,
                  talib.CDLSHORTLINE,
                  talib.CDLSPINNINGTOP,
                  talib.CDLSTALLEDPATTERN,
                  talib.CDLSTICKSANDWICH,
                  talib.CDLTAKURI,
                  talib.CDLTASUKIGAP,
                  talib.CDLTHRUSTING,
                  talib.CDLTRISTAR,
                  talib.CDLUNIQUE3RIVER,
                  talib.CDLUPSIDEGAP2CROWS,
                  talib.CDLXSIDEGAP3METHODS
            ]

            for f in fs:
                result = f(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
                resp = np.where(result == 100)[0][:]
                resn = np.where(result == -100)[0][:]

                if len(resp) != 0:
                    for i in range(len(resp)):
                        if i + 5 < len(df['Close']):
                            op[resp[i: i + 3]] += -1

                if len(resn) != 0:
                    for i in range(len(resn)):
                        if i + 5 < len(df['Close']):
                            op[resn[i: i + 3]] += 1


            for i in range(-7, 8):
                res[i + 7] += len(np.where(op == i)[0][:])


            vals = df['Close'].values
            amt = np.zeros(len(df['Close']))
            gain = np.zeros(len(df['Close']))
            day_diff = np.zeros(len(df['Close'])).astype(int)

            buy_signal = 4
            sell_signal = -4
            for i in range(len(op)):
                if op[i] > buy_signal:
                    for j in range(i + 1, len(op)):
                        if op[j] < sell_signal:
                            amt[i] = vals[j] - vals[i]
                            gain[i] = (vals[j] - vals[i]) * 100 / vals[i]
                            day_diff[i] = j - i
                            break

            df['Amt'] = amt[:]
            df['Gain'] = gain[:]
            df['Day_Diff'] = day_diff[:]

            total_res[0] += len(df['Gain'])
            total_res[1] += len(df['Gain'][df['Gain'] > 0])
            total_res[2] += len(df['Gain'][df['Gain'] < 0])
            total_res[3] += df['Gain'][df['Gain'] != 0].sum()
            total_res[4] += df['Gain'][df['Gain'] > 0].sum()
            total_res[5] += df['Gain'][df['Gain'] < 0].sum()
            total_res[6] += df['Day_Diff'][df['Gain'] != 0].sum()
            total_res[7] += df['Day_Diff'][df['Gain'] > 0].sum()
            total_res[8] += df['Day_Diff'][df['Gain'] < 0].sum()
            total_res[9] += len(np.where(op > buy_signal))


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


            df['ticker'] = ticker
            df.reset_index(inplace=True)
            df.set_index(['ticker', 'Date'], inplace=True)

            pos.append(df[df['Gain'] > 0])
            neg.append(df[df['Gain'] < 0])


        except Exception as e:
            print(e)


compile_data()

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