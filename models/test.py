import tensorflow as tf

x = tf.random.normal(shape=(64, 32, 32, 16))
print(x.shape)