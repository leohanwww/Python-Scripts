# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:18:31 2019

@author: leohanwww
"""

import tensorflow as tf

a = tf.Variable(tf.random_normal([4,4], stddev=0.1))
b = tf.Variable(tf.random_normal([4,4], stddev=0.1))
c = tf.Variable(tf.random_normal([1,2,3,4], stddev=0.1))
concated = tf.concat([a,b],1)
diag = tf.constant([1,3,3,4])
tran = tf.transpose(c)

#reversed = tf.reverse(c,-1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print(sess.run(a))
    #print(sess.run(b))
    #print(sess.run(concated))
    #print(sess.run(tf.shape(concated)))
    #print(sess.run(reversed))
    #print(sess.run(tf.diag(diag)))
    #print(sess.run(tf.shape(tran)))
    #tf.train.Saver.save(sess, save_path='d:/model')