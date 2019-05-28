import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1, 2])
    b = tf.constant([3, 4])
with tf.device('/gpu:0'):
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))