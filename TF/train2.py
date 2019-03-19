import tensorflow as tf


matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])

product=tf.matmul(matrix1,matrix2)

#method1
sess=tf.Session()
result=sess.run(product)
print(sess.run(matrix1))
print(sess.run(matrix2))
print(result)
sess.close()

#method2(no need to close)
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)
