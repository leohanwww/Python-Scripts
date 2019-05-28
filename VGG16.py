'''
使用数据增强来使用训练过的conv2d模型,在模型后端自行增加分类层,进行猫-狗2分类
'''

from keras.applications import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
import os


base_dir = 'd:/dogs-vs-cats/small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

conv_base = VGG16(  # 这个模型到卷积层结束已经训练过并包含了权重参数
    weights='imagenet',  # 模型初始化的权重检查点
    include_top=False,  # 不包含密集连接分类器,使用自定义分类器
    input_shape=((150, 150, 3))  # 不指定输入大小,网络将能处理任意形状的输入
)

model = models.Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print('trainable weights before freezing:', len(model.trainable_weights))
conv_base.trainable = False  # 不更新卷积层的参数
print('trainable weights after freezing:', len(model.trainable_weights))

# 训练数据生成器,使用数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 测试数据生成器
test_datagen = ImageDataGenerator(rescale=1./255)

# 利用生成器,从文件夹中的图片生成测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

validation_datagen = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)

# 此处从生成器中产生的数据训练
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 每个epoch从generator产生100次数据
    epochs=30,
    validation_data=validation_datagen,
    validation_steps=50
)

model.save('cats-vs-dogs_small_3.h5')

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation acc')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()