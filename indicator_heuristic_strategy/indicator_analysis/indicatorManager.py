import pickle
import talib
import pandas as pd
import numpy as np
import csv

# Check the answer https://stackoverflow.com/a/20627316/5512020
# to prevent the warning: A value is trying to be set on a copy of a slice from a DataFrame.
pd.options.mode.chained_assignment = None  # default='warn'

total_res = np.zeros(13)

table_name = 'stock_data'

with open('../../sp500tickers.pickle', 'rb') as f:
    tickers = pickle.load(f)

with open('../data.pickle', 'rb') as f:
    df_total = pickle.load(f)

def analysePrevDays(prevCloses):
    plusCount = 0
    for i in range(1, len(prevCloses)):
        if prevCloses[i] > prevCloses[i - 1]:
            plusCount += 1

    return plusCount

def analyze_indicator(indicator, max_buy_val, min_sell_val, period, needs_HL):
    # without count the ticker name is incorrect
    for count, ticker in enumerate(tickers):
        # if ticker == 'AAP':
            df = df_total[df_total['Ticker'] == ticker]
            df.reset_index(inplace=True)
            df.set_index(['Ticker', 'Date'], inplace=True)
            df.drop(['index'], axis=1, inplace=True)

            if not needs_HL:
                df[indicator] = getattr(talib, indicator)(df['Close'].values, timeperiod=period)
            else:
                df[indicator] = getattr(talib, indicator)(df['High'].values, df['Low'].values, df['Close'].values,
                                                          timeperiod=period)

            df['Res'] = np.where(df[indicator] < max_buy_val, 1, np.where(df[indicator] > min_sell_val, -1, 0))
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

            sl = open('sl.csv', 'a', newline='')
            writer = csv.writer(sl)

            for k in range(len(op)):
                if op[k] > buy_signal and anaV[k] == 1 and k > closePosition:
                    for i in range(k + 3, k + 23):
                        if i - k == 3 or i - k == 4:
                            plusCount = analysePrevDays(vals[k:i])
                        else:
                            plusCount = analysePrevDays(vals[i-5:i])

                        if plusCount >= 3:
                            sl = False
                            priceDiffRow = []
                            for j in range(i + 1, len(op)):
                                change = (vals[j] - vals[i]) * 100 / vals[i]
                                priceDiffRow.append(change)

                                if change < - 10 and not sl:
                                    sl = True

                                if op[j] < sell_signal:
                                    amt[i] = vals[j] - vals[i]
                                    gain[i] = (vals[j] - vals[i]) * 100 / vals[i]
                                    day_diff[i] = j - i
                                    sell_days[j] += 1
                                    closePosition = j
                                    if sl:
                                        if gain[i] > 0:
                                            total_res[11] += 1
                                        else:
                                            total_res[12] += 1

                                        writer.writerow(priceDiffRow)
                                    break

                        if closePosition > i:
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

    total_res[3] = total_res[3] / total_res[9]
    total_res[4] = total_res[4] / total_res[1]
    total_res[5] = total_res[5] / total_res[2]
    total_res[6] = total_res[6] / total_res[9]
    total_res[7] = total_res[7] / total_res[1]
    total_res[8] = total_res[8] / total_res[2]

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

    row = [indicator, max_buy_val, min_sell_val, period]
    # check for diff append and extend https://stackoverflow.com/a/252711/5512020
    row.extend([element for element in total_res])
    csvRow = ','.join(map(str, row))
    with open('analysis.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csvRow)
    f.close()


analyze_indicator(
    indicator='RSI',
    max_buy_val=25,
    min_sell_val=65,
    period=14,
    needs_HL=False
)

