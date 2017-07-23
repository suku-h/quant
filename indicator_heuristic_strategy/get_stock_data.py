import sqlite3
import pickle
import pandas as pd
import os
import pandas_datareader as pdr
import datetime as dt

conn = sqlite3.connect("data.db")
cur = conn.cursor()
table_name = 'stock_data'
cur.execute("DROP TABLE IF EXISTS '" + table_name + "'")
cur.execute("CREATE TABLE '" + table_name + "'(Date date, Ticker text, Open double, High double, Low double, Close double, Volume integer)")
conn.commit()

def get_stock_data():
    with open('../sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2016, 12, 31)

    for ticker in tickers:
        if not os.path.exists("F:/data/sp500/{}.csv".format(ticker)):
            try:
                df = pdr.DataReader(ticker, 'google', start, end)
                df.to_csv("F:/data/sp500/{}.csv".format(ticker))
            except:
                print("Couldn't find", ticker)

    # without count the ticker name is incorrect
    for count, ticker in enumerate(tickers):
        print(count, ticker)
        filepath = 'F:/data/sp500/{}.csv'.format(ticker)
        df = pd.read_csv(filepath)
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        ratio = 1
        try:
            ratio = df['Close'] / df['Adj Close']
            df['Close'] = df['Adj Close']
            df.drop(['Adj Close'], axis=1, inplace=True)
        except:
            pass

        df['Open'] = df['Open'] / ratio
        df['High'] = df['High'] / ratio
        df['Low'] = df['Low'] / ratio
        df['Volume'] = round(df['Volume'] / ratio)
        df['Ticker'] = ticker

        df.to_sql(table_name, conn, if_exists='append')

    df_total = pd.read_sql_query("SELECT Date, Ticker, Open, High, Low, Close, Volume FROM '" + table_name + "'", conn)
    df_total['Date'] = pd.to_datetime(df_total['Date']).apply(lambda x: x.date())
    df_total.to_pickle('data.pickle')
    # remember to close connection
    conn.close()


get_stock_data()
