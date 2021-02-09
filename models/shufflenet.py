"""
ShuffleNet in tensorflow 2

Xiangyu Zhang, ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
https://arxiv.org/pdf/1707.01083
"""

import tensorflow as tf


class ShuffleLayer(tf.keras.layers.Layer):
    def __init__(self, g):
        super(ShuffleLayer, self).__init__()
        self.g = g

    def call(self, x):
        b, h, w, c = tf.shape(x)
        x = tf.reshape(x, [b, h, w, self.g, c//self.g])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [b, h, w, c])
        return x


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(Bottleneck, self).__init__()
        self.filters = filters
        self.strides = strides
        self.groups = groups
        self.expansion = 4
    
    def build(self, input_shape):
        if self.strides > 1 or input_shape[-1] != self.filters * self.expansion:
            self.shortcut = tf.keras.layers.Conv2D(self.filters * self.expansion, 1, self.strides, use_bias=False)
        else:
            self.shortcut = tf.identity

        self.conv1 = tf.keras.layers.Conv2D(self.filters, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
    
        self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, self.strides, 'same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(self.filters * self.expansion, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
    
    def call(self, x):
        res = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += res
        x = self.relu3(x)
        return x
