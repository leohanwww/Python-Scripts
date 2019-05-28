from keras.layers import Embedding, Flatten, Dense
from keras import preprocessing
from keras.datasets import imdb
from keras.models import Sequential

max_features = 1000
max_length = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)

'''
Embedding 层的输入是一个二维整数张量，其形状为(samples, sequence_length)，
每个元素是一个整数序列。它能够嵌入长度可变的序列，例如，对于前一个例子中的
Embedding 层，你可以输入形状为(32, 10)（32 个长度为10 的序列组成的批量）
这个Embedding 层返回一个形状为(samples, sequence_length, embedding_
dimensionality) 的三维浮点数张量
embedding层可以理解为字典,输入整数索引,在内部字典查找整数,输出密集向量
'''

model = Sequential()
model.add(Embedding(1000, 8, input_length=max_length))
# 第一参数是最大单词索引+1,第二参数是嵌入维度
# input_shape=(None, 20) output_shape=(None, 20, 8)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

print(history.history['acc'])