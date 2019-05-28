# -*- coding: utf-8 -*-

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)  # 保留最常出现的10000个单词

# 整个过程是整理电影评论为文本的过程
# {'fawn': 34701, 'tsukino': 52006, 'nunnery': 52007....
# 是一个单词映射整数的字典
word_index = imdb.get_word_index()
# 将字典里的整数和单词翻转,将整数作为key
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reversed_word_index.get(i - 3, '?') for i in train_data[0]]
)
# 将整数序列的评论转换成文字,减去3是因为0,1,2是padding\start of sequence\unknow保留的索引
# ?号是把不能转换的用*代替


# 向量化输入
def vectorize_sequence(sequences, dimension=10000):
    # 创建一个形状为len(sequence), dimension的零矩阵,这样one-hot标志
    out_put = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        out_put[i, sequence] = 1  # 设为有效数据
    return out_put


x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

y_train = train_labels.astype('float32')
y_test = test_labels.astype('float32')

input_tensor = Input(shape=(10000,))
hidden_layer_1 = Dense(16, activation='relu')(input_tensor)
hidden_layer_2 = Dense(16, activation='relu')(hidden_layer_1)
out_layer = Dense(1, activation='sigmoid')(hidden_layer_2)

model = Model(inputs=input_tensor, outputs=out_layer)

model.compile(
    optimizer='rmsprop',
    loss='mse',
    metrics=['accuracy']
)

history = model.fit(
    x_train[10000:],
    y_train[10000:],
    epochs=5,
    batch_size=512,
    validation_data=(x_train[:10000], y_train[:10000])
)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  # 用于显示图例
plt.show()