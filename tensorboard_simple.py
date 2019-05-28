# -*- coding: utf-8 -*-

import tensorflow as tf

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(train_steps):
        xs, ys = mnist.train.next_batch(batch_size)
        if i % 1000 == 0:
            run_options = tf.RunOptions(#运行时需要记录的信息
                trace_level=tf.RunOptions.FULL_TRACE
            )
            run_metadata = tf.RunMetadata()#运行时记录信息的proto
            _, loss_value, step = sess.run(
                [train_op, loss, global_steps], 
                feed_dict={x:xs, y_:ys},
                options=run_options, run_metadata=run_metadata)
            #将节点运行时的信息写入日志文件
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            print(step, loss_value)
        else:
            _, loss_value, step = sess.run(
                [train_op, loss, global_steps], 
                feed_dict={x:xs, y_:ys})