"""
DenseNet in tensorflow 2

Gao Huang, Densely Connected Convolutional Networks
https://arxiv.org/abs/1608.06993
"""
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
        self.conv = tf.keras.layers.Conv2D(filters, 1, 1, use_bias=False)
        self.pool = tf.keras.layers.AvgPool2D()
    
    def call(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


def DenseNet(cfg, input_shape=(32, 32, 3), output_shape=10):
    filters = cfg['growth_rate'] * 2
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', use_bias=False)(inputs)

    for nb in cfg['num_block']:
        for _ in range(nb):
            x = Bottleneck(cfg['growth_rate'])(x)
            filters += cfg['growth_rate']
        filters //= 2
        x = Transition(filters)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)



def DenseNet121():
    cfg = {
        'growth_rate': 32,
        'num_block': [6, 12, 24, 16]
    }
    return DenseNet(cfg)

def DenseNet169():
    cfg = {
        'growth_rate': 32,
        'num_block': [6, 12, 32, 32]
    }
    return DenseNet(cfg)

def DenseNet201():
    cfg = {
        'growth_rate': 32,
        'num_block': [6, 12, 48, 32]
    }
    return DenseNet(cfg)

def DenseNet264():
    cfg = {
        'growth_rate': 32,
        'num_block': [6, 12, 64, 48]
    }
    return DenseNet(cfg)