# -*- coding: utf-8 -*-

from keras.datasets import boston_housing
from keras import models
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 洗数据 先减去每列平均值 再除以每列标准差 得到特征平均值为0, 标准差为1
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# 定义模型, 需要将同一模型多次实例化
def build_model():
    model = models.Sequential()
    model.add(Dense(64, activation='relu',
                    input_shape=(train_data.shape[1], )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# k折验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 80
all_mae_histories = []

for i in range(k):
    print('processing fold : ', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    parital_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0
    )

    parital_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model()

    history = model.fit(
        parital_train_data,
        parital_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=1,
        verbose=0  # 静默模式
    )

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()