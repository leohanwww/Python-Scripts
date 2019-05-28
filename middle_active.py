import matplotlib.pyplot as plt
import numpy as np
from keras import models
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K

model = load_model(
    'C:/Users/leohanwww/Documents/Python Scripts/cats_and_dogs_small_2.h5')

# 主备一张没学习过的照片
img_path = 'D:/dogs-vs-cats/all_images/dog.7525.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
# print('before:', img_tensor.shape)
img_tensor = np.expand_dims(img_tensor, axis=0)  # 需要扩展(增加)维度
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()

# print('after:', img_tensor.shape)

# 选择model前8层,里面有卷积和池化
output_layers = [layer.output for layer in model.layers[:8]]
# 创建一个模型,还是原输入,返回前8层的激活值
activation_model = models.Model(inputs=model.input, outputs=output_layers)

# 返回8个numpy数组组成的列表,每个层激活对应一个numpy数组
activations = activation_model.predict(img_tensor)
# 第一个卷积层输出的激活
first_layer_activation = activations[0]
# print(first_layer_activation.shape)  # 输出是[1, 148, 148, 32],有32个通道

plt.matshow(first_layer_activation[0, :, :, 0])  # 输出第0通道
plt.show()
