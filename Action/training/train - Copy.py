# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from keras.layers.convolutional import Conv1D
from matplotlib import pyplot
import pandas as pd
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load data
    raw_data = pd.read_csv('origin_data.csv', header=0)
    dataset = raw_data.values
    # X = dataset[:, 0:36].astype(float)
    # Y = dataset[:, 36]
    X = dataset[0:7541, 0:51].astype(float)  # 忽略run数据
    Y = dataset[0:7541, 51]

    encoder_Y = [0]*3663 + [1]*3766 + [2]*111
    # one hot 编码
    dummy_Y = np_utils.to_categorical(encoder_Y)
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.25, random_state=9)
    return X_train, Y_train, X_test, Y_test

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    # define model
    print(testy)
    verbose, epochs, batch_size = 0, 180, 1
    n_input=1
    n_feature=51
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    print(trainX.shape)
    # define model
    model = Sequential()
    model.add(LSTM(200, batch_input_shape=(n_input, n_feature,1),activation='relu', stateful=True))
    model.add(Dropout(0.15))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse'])
    # create and fit the LSTM network

    # fit network
    print('Traing.....')
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

# run the experiment
run_experiment()