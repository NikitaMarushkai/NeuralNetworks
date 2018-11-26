import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

output = tf.multiply(x, y)

with tf.Session() as sess:
    result = sess.run(output, feed_dict={x: 2, y: 3})

print(result)
