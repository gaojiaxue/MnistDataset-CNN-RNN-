import tensorflow as tf
import numpy as np
#Save to file
#when restore the dtype and shape must be the same
# W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

# init=tf.initialize_all_variables()

# saver=tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(init)
#     save_path=saver.save(sess,"my_net/save_net.ckpt")
#     print("Save to path:",save_path)


#Restore varible from save_net.ckpt
W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')


#when restore noneed to init

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weigths:",sess.run(W))
    print("biases:",sess.run(b))