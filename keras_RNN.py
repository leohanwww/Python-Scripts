# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_len = 80
max_features = 20000
batch_size = 32

(trainX, trainY), (testX, testY) = imdb.load_data(num_words=max_len)
print(len(trainX), 'train sequences')
print(len(testX), 'test sequences')

trainX = sequence.pad_sequences(trainX, maxlen=max_len)
testX = sequence.pad_sequences(testX, maxlen=max_len)

print('trainX shape:', trainX.shape)
print('testX shape:', testX.shape)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(trainX, trainY,
            batch_size=batch_size,
            epochs=5,
            validation_data=(testX, testY))

score = model.evaluate(testX, testY, batch_size=batch_size)
print('test loss:')
print(score[0])
print('test accuracy:')
print(score[1])