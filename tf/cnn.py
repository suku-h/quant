import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split


with open("../stocks.pickle", 'rb') as f:
    data = pickle.load(f)

with open("../sp500tickers.pickle", 'rb') as f:
    tickers = pickle.load(f)

def process_data(ticker):
    data_days = 20
    pred_days = 15
    df = data[ticker]

    df['ticker'] = ticker
    # need to reset the index to set both ticker and date as index
    df.reset_index(inplace=True)
    df.set_index(['ticker', 'Date'], inplace = True)
    df.fillna(0, inplace=True)

    for i in range(1, data_days + 1):
        # the -i will give the future 1 day data
        df['{}d RSI'.format(i)] = df['RSI'].shift(i)
        df['{}d ADX'.format(i)] = df['ADX'].shift(i)

    for i in range(1, pred_days + 1):
        # the -i will give the future 1 day data
        df['{}d pred'.format(i)] = (df['Close'].shift(-i) - df['Close']) / df['Close']

    for i in range(1,4):
        df['{}d'.format(i)] = (df['Close'].shift(-i) - df['Close']) / df['Close']

    df.dropna()
    return df


def buy_sell_hold(*args):
    cols = [c for c in args]
    # if stock changes by 2% in 7 days then sell /buy
    requirement = 0.05
    for col in cols:
        if col > requirement:
            return 1
        elif col < - 0.05:
            return 0
    return 0


def extract_featuresets(ticker):
    df = process_data(ticker)
    pred_days = 15
    df['target'] = list(map(buy_sell_hold, *[df['{}d pred'.format(i)] for i in range(1, pred_days + 1)]))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    dropcols = ['Close', 'Volume', 'MACD', 'MACD_sign', 'MACD_diff', 'BBMean', 'BBUp', 'BBDown']
    for i in range(1, pred_days + 1):
        dropcols.append('{}d pred'.format(i))
    df.drop(dropcols, axis=1, inplace=True)
    return df


def merge_data():
    total_df = []

    for count, ticker in enumerate(tickers):
        try:
            df = extract_featuresets(ticker)
            # appending to list is faster than appending dataframes
            total_df.append(df)
            print(ticker)

        except Exception as e:
            print(e)


    total_df = pd.concat(total_df, axis=0)
    # ValueError: The truth value of a Series is ambiguous if 'and' 'or' is used
    # for some reason appends adds all date rows to each stock
    # hence if a stock is not listed then it all comes as 0
    # this also removes the stocks with low movement
    total_df = total_df[(total_df['1d'] != 0) & (total_df['2d'] != 0) & (total_df['3d'] != 0)]
    ydf = total_df['target']
    xdf = total_df.drop(['target', '1d', '2d', '3d'], axis = 1)
    xdf.to_pickle('xdf.pickle')
    ydf.to_pickle('ydf.pickle')
    vals = ydf.values.tolist()
    str_vals = [str(i) for i in vals]
    print("Data spread:", Counter(str_vals))


n_nodes_hl1 = 250
n_nodes_hl2 = 250
n_classes = 2
batch_size = 1000

with open("xdf.pickle", 'rb') as f:
    features = pickle.load(f)
with open("ydf.pickle", 'rb') as f:
    labels = pickle.load(f)


def neural_network_model(data, feature_count):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([feature_count, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.transpose(tf.matmul(l2, output_layer['weights']) + output_layer['biases'])
    return output


def train_neural_network(x, y, features, labels):

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    print(len(y_train), len(y_train[y_train == 1]), len(y_train[y_train == 0]))

    # oversampling
    # ypos = y_train[y_train == 1]
    # xpos = X_train[y_train == 1]
    # X_train = pd.concat([xpos, X_train])
    # y_train = pd.concat([ypos, y_train])]

    # undersampling
    ypos = y_train[y_train == 0]
    ypos = ypos[len(ypos)//2 :]
    xpos = X_train[y_train == 0]
    xpos = xpos[len(xpos)//2 :]
    y_train = y_train.loc[y_train.index.difference(ypos.index)]
    X_train = X_train.loc[X_train.index.difference(xpos.index)]
    print(len(y_train), len(y_train[y_train == 1]), len(y_train[y_train == 0]))

    prediction = neural_network_model(x, len(features.columns))

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.transpose(prediction), labels=tf.cast(y, tf.int64)))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 8

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0

            for i in range(int(len(X_train) / batch_size)):
                epoch_x = X_train[i*batch_size: min((i + 1)*batch_size, len(X_train))]
                epoch_y = y_train[i*batch_size: min((i + 1)*batch_size, len(y_train))]
                i, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c

            print('Epoch', epoch + 1, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)

        # InvalidArgumentError (see above for traceback): Expected dimension in the range [-1, 1), but got 1
        # hence don't do argmax for y
        # need to cast prefiction to float
        x_argmax = tf.argmax(tf.transpose(prediction), 1)
        y_argmax = tf.cast(y, tf.int64)
        correct = tf.equal(x_argmax, y_argmax)

        accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))

        print('Accuracy : ', sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

        y1 = y_test[y_test == 1]
        X1 = X_test[y_test == 1]
        print('Accuracy 1: ', sess.run(accuracy, feed_dict={x: X1, y: y1}))

        y0 = y_test[y_test == 0]
        X0 = X_test[y_test == 0]
        print('Accuracy 0: ', sess.run(accuracy, feed_dict={x: X0, y: y0}))


x = tf.placeholder(tf.float32, [None, len(features.columns)])
y = tf.placeholder(tf.float32)
train_neural_network(x, y, features, labels)

