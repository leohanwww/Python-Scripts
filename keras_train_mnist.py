# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense

num_classes = 10

# 输入数据整理
(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX.reshape(trainX.shape[0], 784)
testX = testX.reshape(testX.shape[0], 784)

trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# 标签分类
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)# shape=(60000, 10)

model = Sequential()
model.add(Dense(500, input_shape=(784,), activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, \
    optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=8, epochs=10, \
    validation_data=(testX, testY))

score = model.evaluate(testX, testY)

print('test loss:', score[0])
print('test accuracy:', score[1])