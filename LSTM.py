'''
允许过去的信息稍后进入输入
LSTM保存信息给后面使用,防止较早信息在处理过程中消失
'''

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

max_features = 10000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# (25000,)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)  # (25000, 500)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2  # 划分0.2为验证数据
)
# print(model.summary())