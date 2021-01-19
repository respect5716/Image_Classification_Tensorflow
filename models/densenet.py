import numpy as np
import tensorflow as tf


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(growth_rate*4, 1, 1, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(growth_rate, 3, 1, 'same', use_bias=False)
    
    def call(self, x):
        res = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = tf.concat([res, x], axis=-1)
        return x

class Transition(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Transition, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters, 1,1, use_bias=False)
        self.pool = tf.keras.layers.AvgPool2D()
    
    def call(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))

def create_densenet(num_blocks, growth_rate):
    filters = growth_rate * 2
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', use_bias=False)(inputs)

    for n in num_blocks:
        for _ in range(n):
            x = Bottleneck(growth_rate)(x)
            filters += growth_rate

        filters //= 2
        x = Transition(filters)(x)

    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def DenseNet121():
    return create_densenet([6, 12, 24, 16], 32)

def DenseNet169():
    return create_densenet([6, 12, 32, 32], 32)

def DenseNet201():
    return create_densenet([6, 12, 48, 32], 32)

def DenseNet264():
    return create_densenet([6, 12, 64, 48], 32)

def DenseNetCifar():
    return create_densenet([6, 12, 24, 16], 12)