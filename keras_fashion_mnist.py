

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
plt.imshow(train_images[0])
plt.show()
'''

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('test accuracy:', test_accuracy)

predictions = model.predict(test_images)


pre_cpunt = 0
for i in range(len(predictions)):
    if np.argmax(predictions[i]) == test_labels[i]:
        pre_cpunt += 1
print(pre_cpunt)