import pickle
import talib
import pandas as pd
import numpy as np

# Check the answer https://stackoverflow.com/a/20627316/5512020
# to prevent the warning: A value is trying to be set on a copy of a slice from a DataFrame.
pd.options.mode.chained_assignment = None  # default='warn'

res = np.zeros(15).astype(int)
total_res = np.zeros(11)

table_name = 'stock_data'

def compile_data():
    with open('../../sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    with open('../data.pickle', 'rb') as f:
        df_total = pickle.load(f)

    # without count the ticker name is incorrect
    for count, ticker in enumerate(tickers):
        # if ticker == 'AAP':
            df = df_total[df_total['Ticker'] == ticker]
            df.reset_index(inplace=True)
            df.set_index(['Ticker', 'Date'], inplace=True)
            df.drop(['index'], axis=1, inplace=True)

            df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
            df['Res'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 60, -1, 0))
            df['maV'] = df['Volume'].rolling(100).mean()
            df['anaV'] = np.where(df['Volume'] > 1.25 * df['maV'], 1, 0)

            op = df['Res'].values
            vals = df['Close'].values
            amt = np.zeros(len(df['Res']))
            gain = np.zeros(len(df['Res']))
            day_diff = np.zeros(len(df['Res'])).astype(int)
            sell_days = np.zeros(len(df['Res'])).astype(int)
            anaV = df['anaV'].values

            buy_signal = 0
            sell_signal = 0
            closePosition = -1
            countBuy = False
            for i in range(len(op)):
                if op[i] > buy_signal and anaV[i] == 1 and i > closePosition:
                    if not countBuy:
                        countBuy = True
                    else:
                        for j in range(i + 1, len(op)):
                            if op[j] < sell_signal:
                                amt[i] = vals[j] - vals[i]
                                gain[i] = (vals[j] - vals[i]) * 100 / vals[i]
                                day_diff[i] = j - i
                                sell_days[j] += 1
                                closePosition = j
                                countBuy = False
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


            if ticker == 'AAP':
                print(df.tail(80))
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

            df['p'] = np.where(df['Res'] > buy_signal, 1, 0)
            df['n'] = np.where(df['Sell_Days'] > 0, 1, 0)


compile_data()

print(res)
print('Total days', total_res[0])
print('Total buys', total_res[9])
print('Total sell days', total_res[10])
print('+', total_res[1])
print('-', total_res[2])
print('avg gain', total_res[3] / total_res[9])
print('avg + gain', total_res[4] / total_res[1])
print('avg - gain', total_res[5] / total_res[2])
print('avg day diff', total_res[6] / total_res[9])
print('avg + day diff', total_res[7] / total_res[1])
print('avg - day diff', total_res[8] / total_res[2])