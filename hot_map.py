from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = ''

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)  # 转换为(224,224,3)的数组

x = np.expand_dims(x, axis=0)  # 添加维度

x = preprocess_input(x)

pred = model.predict(x)

>>> preds = model.predict(x)
>>> print('Predicted:', decode_predictions(preds, top=3)[0])
Predicted:', [(u'n02504458', u'African_elephant', 0.92546833),
(u'n01871265', u'tusker', 0.070257246),
(u'n02504013', u'Indian_elephant', 0.0042589349)]
>>> np.argmax(preds[0])
386

# 预测向量中的'非洲象'元素
african_elephant_output = model.output[:, 386]
# 最后一个卷积层输出的特征图
last_conv_layer = model.get_layer('block5_conv3')
# 非洲象对于conv3输出特征图的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],
[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)