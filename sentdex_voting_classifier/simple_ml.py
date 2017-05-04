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
with open("sp500tickers.pickle", 'rb') as f:
    tickers = pickle.load(f)

def process_data(ticker):
    # predict about next 7 days
    hm_days = 20
    # without mentioning index_col you get an error
    df = pd.read_csv(all_data, index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        # the -i will give the future 1 day data
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

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
    tickers, df = process_data(ticker)
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

    # we have to be very accurate with what features to be sent and not send the columns for values to be predicted
    # normalizing the values
    df_vals = df[[t for t in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], np.nan)
    df_vals.fillna(0, inplace=True)
    print(df_vals.columns)
    x = df_vals.values
    y = df['{}_target'.format(ticker)].values
    print(x)
    print(y)
    return x, y

def do_ml(ticker):
    X, y = extract_featuresets(ticker)
    # without test_size = the line crashes
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)
    print(X_test)
    print(y_test)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    print('Accuracy:', confidence)
    print("Predicted Spread:", Counter(predictions))
    return confidence


def do_time_ml(ticker):
    X, y = extract_featuresets(ticker)
    # without test_size = the line crashes
    tscv = TimeSeriesSplit(n_splits=3)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier()),
                            ('gap', GaussianProcessClassifier()),
                            ('bag', BaggingClassifier()),
                            ('nn', MLPClassifier(max_iter=2000))])
    for train_index, test_index in tscv.split(X):
        print(train_index, test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # need to have () after the classifier otherwise it gives an error
        # TypeError: get_params() missing 1 required positional argument: 'self'
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        predictions = clf.predict(X_test)
        print('Accuracy:', confidence)
        print("Predicted Spread:", Counter(predictions))

do_time_ml('AAPL')