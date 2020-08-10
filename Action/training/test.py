# LSTM for international airline passengers problem with memory
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, Embedding,Activation
from keras.layers import LSTM,GRU ,RNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

look_back = 36
batch_size = 1
# load data
# load data
raw_data = pd.read_csv('data_under_scene.csv', header=0)
dataset = raw_data.values
# X = dataset[:, 0:36].astype(float)
# Y = dataset[:, 36]
X = dataset[0:3329, 0:36].astype(float)  # 忽略run数据
Y = dataset[0:3329, 36]

encoder_Y = [0]*1732 + [1]*797 + [2]*800
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)

X_train = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))
print(X_train.shape)
# create and fit the LSTM network

# model = Sequential()
# model.add(LSTM(100,activation='relu', batch_input_shape=(batch_size, look_back, 1), stateful=True))
# model.add(Dropout(0.15))
# model.add(Dense(1, activation='relu'))
# model.add(Dense(4, activation='softmax'))

step_size = 1

# LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape=(1, look_back), return_sequences = True))
model.add(LSTM(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy']) # Try SGD, adam, adagrad and compare!!!
history=model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=1, verbose=2)
model.save('framewise_recognition_2.h5')

score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['acc', 'val_acc','loss','val_loss'], loc='upper left')
plt.show()


# #model.compile(loss='mean_squared_error', optimizer='adam')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
# _, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
# print(accuracy)
