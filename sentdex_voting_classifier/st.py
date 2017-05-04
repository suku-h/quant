import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
# candlestick_ohlc no longer part of matplotlib
# so download the deprecated package and then use it
from sentdex_voting_classifier.mpl_finance.mpl_finance import candlestick_ohlc
import pandas as pd

style.use('ggplot')
# start = dt.datetime(2000,1,1)
# end = dt.datetime(2016,12,31)
#
# df = pdr.DataReader('TSLA', 'yahoo', start, end)
# df.to_csv('tsla.csv')

df = pd.read_csv("tsla.csv", parse_dates=True, index_col = 0)

# df['Adj Close'].plot()
# min_periods means that it will calculate mean for max available days greater than the min_period
# ie if 10 days of data is available and min_period = 4 then mean of 10 taken but if only 3 days available then NaN
df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()

# resample data to 10 day data.
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
# setting date as index for the new df
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
# first parameter is grid size here 6 rows 1 column
# 2nd param is starting point of the plot
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
# share_x tells that the 2 subplots are linked by common axis
# so a change in any plot (like zoom or selection) will be reflected in the other plot
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex = ax1)
# first param is X axis which is index of df 2nd param is y
# ax1.plot(df.index, df["Adj Close"])
# ax1.plot(df.index, df["100ma"])
# ax2.bar(df.index, df["Volume"])
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()