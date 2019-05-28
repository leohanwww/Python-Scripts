# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:14:01 2019

@author: leohanwww
"""

import tensorflow as tf

with tf.variable_scope('foo'):
    v = tf.get_variable('v', [1], initializer=tf.constant_initializer(0.1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v.shape))