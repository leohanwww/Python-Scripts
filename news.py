# -*- coding: utf-8 -*-

from keras.datasets import reuters
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
      num_words=10000)

word_index = reuters.get_word_index()
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reversed_word_index.get(i - 3, '?') for i in train_data[0]]
)
# print(decoded_review)


# 向量化输入
def vectorize_sequence(sequences, dimension=10000):
    # 创建一个形状为len(sequence), dimension的零矩阵,这样one-hot标志
    out_put = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        out_put[i, sequence] = 1  # 设为有效数据
    return out_put


train_data = vectorize_sequence(train_data)
test_data = vectorize_sequence(test_data)

'''
# one_hot化标签
def one_hot_labels(labels, dimension=46):
    result = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        result[i, label] = 1
    return result


# train_labels = one_hot_labels(train_labels)
# test_labels = one_hot_labels(test_labels)
'''

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 开始构建网络
input_tensor = Input(shape=(10000,))
hidden_layer_1 = Dense(64, activation='relu')(input_tensor)
hidden_layer_2 = Dense(64, activation='relu')(hidden_layer_1)
output_layer = Dense(46, activation='softmax')(hidden_layer_2)

model = Model(inputs=input_tensor, outputs=output_layer)

model.compile(
      optimizer='rmsprop',
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

history = model.fit(
                    train_data[1000:],
                    train_labels[1000:],
                    epochs=10,
                    batch_size=512,
                    validation_data=(train_data[:1000], train_labels[:1000])
)


# 画出图像
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()