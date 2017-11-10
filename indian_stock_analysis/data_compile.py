import pandas as pd
from datetime import datetime as dt
import os
import nsepy
import numpy as np
import time
import glob
import dateutil.parser as dateparser

path =r'F:\data\nse500\indices'
allFiles = glob.glob(path + "\*.csv")
data = pd.concat((pd.read_csv(f, index_col=None, header=0) for f in allFiles))
data.drop_duplicates(inplace=True)

stocks = data['Symbol'].values
for i in range(len(stocks)):
    if stocks[i] == 'FAGBEARING':
        stocks[i] = 'SCHAEFFLER'


def get_files(stock, series, check):
    start = dt(2007, 1, 1)
    end = dt(2017, 11, 10)
    checkVal = 1
    if check:
        os.path.exists('F:/data/nse500/' + series + '/{}.csv'.format(stock))
        checkVal = 0

    if checkVal == 1:
        try:
            df = nsepy.get_history(stock, start, end, series=series)
            # Date is the index
            df.reset_index(inplace=True)
            # 9/7/12 is repeated for several stocks, so delete the repeated entry
            date = df['Date'].values
            dropIndex = []
            for j in range(len(date) - 1):
                if date[j] == date[j + 1]:
                    dropIndex.append(j + 1)

            if len(dropIndex) > 0:
                df.drop(df.index[dropIndex], inplace=True)

            if len(df) > 1:
                df.to_csv('F:/data/nse500/' + series + '/{}.csv'.format(stock), index=False)
        except Exception as e:
            print(e)
            print('Couldn\'t find', series, stock)


def get_data():
    print('getting data')
    if not os.path.exists('F:/data/nse500/ALL'):
        os.makedirs('F:/data/nse500/ALL')

    for i in range(len(stocks)):
        get_files(stocks[i], 'ALL', check=False)

    print('got all data')


def extractCorpActions():
    print('extracting corporate actions')
    df_corp_actions = pd.read_csv('Corporate_Actions.csv')
    df_scrips = pd.read_csv('ListOfScrips.csv')

    # # Select the columns you want for the dictionary
    df_dict = df_scrips[['Security Code', 'ISIN No']]
    dict = df_dict.set_index('Security Code').T.to_dict()

    df_corp_actions['ISIN Code'] = df_corp_actions['Security Code'].map(dict)
    df_corp_actions.dropna(inplace=True)
    isin = df_corp_actions['ISIN Code'].values
    for i in range(len(isin)):
        isin[i] = isin[i]['ISIN No']

    df_corp_actions['ISIN Code'] = isin[:]
    df_corp_actions['actions'] = np.where(df_corp_actions['Purpose'].str.contains('Bonus'), 'Bonus',
                                          np.where((df_corp_actions['Purpose'].str.contains('Dividend')) & (
                                              ~df_corp_actions['Purpose'].str.contains('Dividend'))
                                              , 'Dividend',
                                                   np.where(df_corp_actions['Purpose'].str.contains('Split'), 'Split',
                                                            '')))
    df_corp_actions['temp'] = np.where(df_corp_actions['Purpose'].str.contains('Bonus'),
                                       df_corp_actions['Purpose'].str.replace('Bonus issue ', ''), '')
    df_corp_actions['iniVal'] = np.where(df_corp_actions['temp'].str.len() > 0,
                                         df_corp_actions['temp'].apply(lambda x: x[:x.find(':')]), '')
    df_corp_actions['finVal'] = np.where(df_corp_actions['temp'].str.len() > 0,
                                         df_corp_actions['temp'].apply(lambda x: x[x.find(':') + 1:]), '')

    df_corp_actions['temp'] = np.where(df_corp_actions['Purpose'].str.contains('Dividend'),
                                       df_corp_actions['Purpose'].apply(lambda x: x[x.find('Rs. -') + 6:]), '')
    df_corp_actions['iniVal'] = np.where(df_corp_actions['temp'].str.len() > 0, '0', df_corp_actions['iniVal'])
    df_corp_actions['finVal'] = np.where(df_corp_actions['temp'].str.len() > 0, df_corp_actions['temp'],
                                         df_corp_actions['finVal'])

    df_corp_actions['temp'] = np.where(df_corp_actions['Purpose'].str.contains('Split'),
                                       df_corp_actions['Purpose'].str.replace('Stock  Split From Rs.|/-', ''), '')
    df_corp_actions['temp'] = np.where(df_corp_actions['temp'].str.len() > 0,
                                       df_corp_actions['temp'].str.replace('/- to Rs.| to Rs.', ':'), '')
    df_corp_actions['iniVal'] = np.where(df_corp_actions['temp'].str.len() > 0,
                                         df_corp_actions['temp'].apply(lambda x: x[:x.find(':')]),
                                         df_corp_actions['iniVal'])
    df_corp_actions['finVal'] = np.where(df_corp_actions['temp'].str.len() > 0,
                                         df_corp_actions['temp'].apply(lambda x: x[x.find(':') + 1:]),
                                         df_corp_actions['finVal'])

    df_corp_actions.drop('temp', axis = 1)

    print('got corporate actions')

    return df_corp_actions


def get_undownloaded_stocks():
    for i in range(len(stocks)):
        if not os.path.exists('F:/data/nse500/ALL/' + stocks[i] + '.csv'):
            get_files(stocks[i], 'ALL', False)


def isNaN(df):
    # One of the properties of NaN is that NaN != NaN is True.
    return df != df


def addNonTradedPrices():
    print('adding non traed prices')
    if not os.path.exists('F:/data/nse500/adjusted'):
        os.makedirs('F:/data/nse500/adjusted')

    for i in range(len(stocks)):
        refdf = pd.read_csv('F:/data/nse500/all/RELIANCE.csv')
        newdf = refdf['Date']
        try:
            df = pd.read_csv('F:/data/nse500/all/' + stocks[i] + '.csv')
            refIndex = newdf[newdf == df['Date'].iloc[0]].index.tolist()[0]

            newdf = newdf[newdf.index >= refIndex].to_frame()
            # this will just copy the values of df['open']
            # and leave NaN if len(newdf['Open']) < len(df['Open'])
            # newdf['Open'] = df['Open']
            newdf = pd.merge(newdf,df, on=['Date'], how='outer')
            newdf['Open'] = np.where(isNaN(newdf['Open']), newdf['Close'].shift(1), newdf['Open'])
            newdf['High'] = np.where(isNaN(newdf['High']), newdf['Close'].shift(1), newdf['High'])
            newdf['Low'] = np.where(isNaN(newdf['Low']), newdf['Close'].shift(1), newdf['Low'])
            newdf['Close'] = np.where(isNaN(newdf['Close']), newdf['Close'].shift(1), newdf['Close'])
            newdf['Last'] = np.where(isNaN(newdf['Last']), newdf['Close'].shift(1), newdf['Last'])
            newdf['VWAP'] = np.where(isNaN(newdf['VWAP']), newdf['Close'].shift(1), newdf['VWAP'])
            newdf['Volume'] = np.where(isNaN(newdf['Volume']), 0, newdf['Volume'])
            newdf['Symbol'] = np.where(isNaN(newdf['Symbol']), newdf['Symbol'].shift(1), newdf['Symbol'])
            newdf['Turnover'] = np.where(isNaN(newdf['Turnover']), newdf['Turnover'].shift(1), newdf['Turnover'])
            newdf['Trades'] = np.where(isNaN(newdf['Trades']), 0, newdf['Trades'])

            if 'Unnamed: 0' in newdf.columns:
                newdf.drop('Unnamed: 0', axis = 1, inplace = True)

            newdf.to_csv('F:/data/nse500/adjusted/{}.csv'.format(stocks[i]), index=False, index_label=None)
        except:
            print(stocks[i])

    print('added not traded prices')


def adjustDividend(date, dividend, df):
    df['Open'] = np.where(df['sec'] <= date, df['Open'] - dividend, df['Open'])
    df['High'] = np.where(df['sec'] < date, df['High'] - dividend, df['High'])
    df['Low'] = np.where(df['sec'] < date, df['Low'] - dividend, df['Low'])
    df['Last'] = np.where(df['sec'] < date, df['Last'] - dividend, df['Last'])
    df['Close'] = np.where(df['Close'] < date, df['Close'] - dividend, df['Close'])
    df['VWAP'] = np.where(df['sec'] < date, df['VWAP'] - dividend, df['VWAP'])

    return df

def adjustBonusSplit(date, ratio, df):
    df['Open'] = np.where(df['sec'] <= date, df['Open'] * ratio, df['Open'])
    df['High'] = np.where(df['sec'] < date, df['High'] * ratio, df['High'])
    df['Low'] = np.where(df['sec'] < date, df['Low'] * ratio, df['Low'])
    df['Last'] = np.where(df['sec'] < date, df['Last'] * ratio, df['Last'])
    df['Close'] = np.where(df['Close'] < date, df['Close'] * ratio, df['Close'])
    df['VWAP'] = np.where(df['sec'] < date, df['VWAP'] * ratio, df['VWAP'])
    df['Volume'] = np.where(df['sec'] < date, df['Volume'] * ratio, df['Volume'])

    return df


def adjustData():
    print('started adjusting data')
    df_corp_actions = extractCorpActions()
    stockMap = data[['Symbol', 'ISIN Code']].values
    for i in range(len(stocks)):
        df_actions = df_corp_actions[(df_corp_actions['ISIN Code'] == stockMap[i][1]) & (df_corp_actions['finVal'] is not None)]
        if len(df_actions) > 0:
            # getting timestamp from a dd-MMM-YYYY date format
            df_actions['sec'] = df_actions['Ex Date'].apply(lambda x: int(time.mktime(dateparser.parse(x).timetuple())))
            actions = df_actions[['sec', 'actions', 'iniVal', 'finVal']].values

            df = pd.read_csv('F:/data/nse500/all/' + stockMap[i][0] + '.csv')
            df['sec'] = df['Date'].apply(lambda x: dt.strptime(x, "%Y-%m-%d").timestamp())

            for j in range(len(actions)):
                if actions[j][1] == 'Dividend':
                    try:
                        df = adjustDividend(actions[j][0], float(actions[j][3]), df)
                    except:
                        print(stocks[i], actions[j][3])

                if actions[j][1] == 'Split':
                    ratio = int(actions[j][3]) / int(actions[j][2])
                    df = adjustBonusSplit(actions[j][0], ratio, df)

                if actions[j][1] == 'Bonus':
                    ratio = (int(actions[j][3]) + int(actions[j][2])) / int(actions[j][2])
                    df = adjustBonusSplit(actions[j][0], ratio, df)

    print('data manipulation complete')


def removeLowVolumePeriods():
    print('removing low volume stocks')
    if not os.path.exists('F:/data/nse500/selected'):
        os.makedirs('F:/data/nse500/selected')

    for i in range(len(stocks)):
        try:
            df = pd.read_csv('F:/data/nse500/all/' + stocks[i] + '.csv')
            df['prod'] = df['Close'] * df['Volume']
            subdf = df[df['prod'] < 400000]
            if len(subdf) > 0:
                # this gives the index of the last entry
                lastLowVolDate = subdf['Date'].iloc[-1]
                lastLowVolDateSeconds = int(time.mktime(dateparser.parse(lastLowVolDate).timetuple()))
                df['sec'] = df['Date'].apply(lambda x: int(time.mktime(dateparser.parse(x).timetuple())))
                selecteddf = df[df['sec'] > lastLowVolDateSeconds]
                selecteddf.drop(['prod', 'sec'], axis = 1, inplace=True)
                selecteddf.set_index('Date')
                selecteddf.to_csv('F:/data/nse500/selected/{}.csv'.format(stocks[i]), index_label=None)
            else:
                df.drop('prod', axis = 1, inplace=True)
                df.to_csv('F:/data/nse500/selected/{}.csv'.format(stocks[i]), index=False, index_label=None)
        except Exception as e:
            print(e)
            print(stocks[i])


    print('finished removing low volume stocks')



get_data()
get_undownloaded_stocks()
extractCorpActions()
addNonTradedPrices()
removeLowVolumePeriods()
adjustData()