# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:09:12 2019

@author: leohanwww
"""

import tensorflow as tf

#v1 = tf.Variable(tf.constant(1.0))
#v2 = tf.Variable(tf.constant(2.0))
#result = v1 + v2
#init_op = tf.global_variables_initializer()
#saver = tf.train.Saver()

'''
with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess,'d:/tensorflow/model/model.ckpt')
    
with tf.Session() as sess:
    saver.restore(sess,'d:/tensorflow/model/model.ckpt')
    print(sess.run(result))
'''
'''
saver = tf.train.import_meta_graph('d:/tensorflow/model/model.ckpt.meta')
with tf.Session() as sess:
	saver.restore(sess,'d:/tensorflow/model/model.ckpt')
	print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))'''
    
v1 = tf.Variable(tf.constant(5.0),name='ov1')
v2 = tf.Variable(tf.constant(9.0),name='ov2')
saver = tf.train.Saver({'v1': v1, 'v2': v2})
   
with tf.Session() as sess:
    saver.restore(sess,'d:/tensorflow/model/model.ckpt')
    print(sess.run(v1.name))