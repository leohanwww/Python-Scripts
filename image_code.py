# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf

PATH = 'D:/tensorflow/flower_photos/daisy'
imgge_data_raw = tf.gfile.FastGFile(PATH, 'rb').read()


with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(imgge_data_raw)
    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    img_data = tf.image.resize_images(img_data, [500, 500], method=0)


    img_data = tf.image.convert_image_dtype(img_data, tf.uint8)
    encode_img = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile('d:/tensorflow/flower_photos/img_re.jpg', 'wb') as f:
      f.write(encode_img.eval())
      plt.imshow(img_data.eval())
      plt.show()