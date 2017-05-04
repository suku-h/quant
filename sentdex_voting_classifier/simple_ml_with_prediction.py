import numpy as np
import pandas as pd
import pickle
from collections import Counter
# cross_validation for training and testing set, neighbour bcoz we are doing kmeans neighbours
# VotingClassifier - because we are going to use many classifiers and vote for the best
# RandomForestClassifier is a type of classifier
from sklearn import svm, neighbors
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier

all_data = 'sp500_joined_closes.csv'
with open("stocks.pickle", 'rb') as f:
    data = pickle.load(f)


def process_data(ticker):
    # predict about next 7 days
    hm_days = 20
    # without mentioning index_col you get an error
    df = data[ticker]
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        # the -i will give the future 1 day data
        df['{}_{}d pred'.format(ticker, i)] = np.round((df['Close'].shift(-i) - df['Close']) / df['Close'], decimals=2)
        df['{}_{}d'.format(ticker, i)] = (df['Close'] - df['Close'].shift(i)) / df['Close'].shift(i)

    df.fillna(0, inplace=True)
    return df

# basic strategy is to invest in stocks that are likely to rise by requirement in the future
def buy_sell_hold(*args):
    cols = [c for c in args]
    # if stock changes by 2% in 7 days then sell /buy
    requirement = 0.1
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    df = process_data(ticker)
    hm_days = 20
    # predict the buy sell or hold for the next 7 days and store it in a map
    # buy_sell_hold returns a 1, or -1 if ANY of the 7 inputs exceeds requirement,
    # not a value for each input. So it returns one output for n inputs.﻿
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)]))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    # Just to see how many buy, sell or hold instructions from the model
    print("Data spread:", Counter(str_vals))
    df.fillna(0, inplace=True)
    # replace infinite changes in prices with np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # print column name of dataframe
    # print(list(df.columns))
    # we have to be very accurate with what features to be sent and not send the columns for values to be predicted
    # normalizing the values
    df['PCT Close'] = df['Close'].pct_change()
    df['PCT Close'] = df['PCT Close'].replace([np.inf, -np.inf], np.nan)
    df['PCT Close'].fillna(0, inplace=True)

    xcols = ['{}_{}d'.format(ticker, i) for i in range(1, hm_days + 1)]
    print(xcols)

    p = df[xcols]

    dropcols = ['Volume', 'MACD', 'MACD_sign', 'MACD_diff', 'BBMean', 'BBUp', 'BBDown', 'PCT Close', 'RSI', 'ADX']
    q = df[dropcols]
    print(q)
    print(p)
    for col in xcols:
        dropcols.append(col)

    z = df[dropcols]
    print(z)
    y = df.drop(dropcols, axis=1, inplace = False)
    print(list(y.columns))

    ypredcols = ['{}_{}d pred'.format(ticker, i) for i in range(1, hm_days + 1)]
    dropcols = ['Volume', 'MACD', 'MACD_sign', 'MACD_diff', 'BBMean', 'BBUp', 'BBDown', '{}_target'.format(ticker)]
    for col in ypredcols:
        dropcols.append(col)

    x = df.drop(dropcols, axis=1, inplace=False)
    print(list(x.columns))

    y.to_csv('y_aes_1020_pred.csv')
    y.drop(['Close'], axis=1, inplace = True)

    x.to_csv('x_aes_1020_pred.csv')
    x.drop(['Close'], axis=1, inplace = True)
    return x, y


def do_ml(ticker):
    X, y = extract_featuresets(ticker)
    # without test_size = the line crashes
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)

    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    print('Accuracy:', confidence)
    print("Predicted Spread:", Counter(predictions))
    return confidence


def do_time_ml(ticker):
    X, y = extract_featuresets(ticker)
    num_splits = 5

    tscv = TimeSeriesSplit(n_splits=num_splits)
    # need to have () after the classifier otherwise it gives an error
    # TypeError: get_params() missing 1 required positional argument: 'self'
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier()),
                            ('gap', GaussianProcessClassifier()),
                            ('bag', BaggingClassifier()),
                            ('nn', MLPClassifier(max_iter=2000))])
    i = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train['{}_target'.format(ticker)])
        confidence = clf.score(X_test, y_test['{}_target'.format(ticker)])
        predictions = clf.predict(X_test)
        print('Accuracy:', confidence)
        print("Predicted Spread:", Counter(predictions))


        clf.fit(X_train, y_train['{}_10d pred'.format(ticker)])
        confidence = clf.score(X_test, y_test['{}_10d pred'.format(ticker)])
        predictions = clf.predict(X_test)
        print('p Accuracy:', confidence)
        print("p Predicted Spread:", Counter(predictions))
        i += 1
        if i == num_splits:
            np.savetxt('p_aes_1020_pred.csv', predictions, delimiter=',')
            np.savetxt('t_aes_1020_pred.csv', X_test, delimiter=',')


do_time_ml('AES')