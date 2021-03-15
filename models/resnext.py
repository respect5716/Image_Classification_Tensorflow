"""
ResNeXt in tensorflow 2

Saining Xie, Aggregated Residual Transformations for Deep Neural Networks
https://arxiv.org/abs/1611.05431
"""

import tensorflow as tf

class Block(tf.keras.layers.Layer):
    def __init__(self, bottleneck_width, cardinality, strides, **kwargs):
        super(Block, self).__init__()
        self.filters = bottleneck_width * cardinality
        self.cardinality = cardinality
        self.strides = strides
        self.expanded_filters = 2 * self.filters
        self.kwargs = kwargs

    def build(self, input_shape):
        if self.strides > 1 or input_shape[-1] != self.expanded_filters:
            self.shortcut = tf.keras.layers.Conv2D(self.expanded_filters, 1, self.strides, use_bias=False, **self.kwargs)
        else:
            self.shortcut = tf.identity

        self.conv1 = tf.keras.layers.Conv2D(self.filters, 1, 1, 'same', use_bias=False, **self.kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
    
        self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, self.strides, 'same', groups=self.cardinality, use_bias=False, **self.kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(self.expanded_filters, 1, 1, 'same', use_bias=False, **self.kwargs)
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

class Stack(tf.keras.layers.Layer):
    def __init__(self, bottleneck_width, cardinality, strides, num_block, **kwargs):
        super(Stack, self).__init__()
        strides = [strides] + [1] * (num_block - 1)
        self.blocks = [Block(bottleneck_width, cardinality, s, **kwargs) for s in strides]
    
    def call(self, x):
        for b in self.blocks:
            x = b(x)
        return x

def ResNeXt(cfg, input_shape=(32, 32, 3), output_shape=10, **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(16, 3, 1, 'same', use_bias=False, **kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = Stack(cfg['bottleneck_width'], cfg['cardinality'], 1, cfg['num_block'], **kwargs)(x)
    x = Stack(cfg['bottleneck_width']*2, cfg['cardinality'], 2, cfg['num_block'], **kwargs)(x)
    x = Stack(cfg['bottleneck_width']*4, cfg['cardinality'], 2, cfg['num_block'], **kwargs)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax', **kwargs)(x)
    return tf.keras.Model(inputs, outputs)

def ResNext29_4x16d(**kwargs):
    cfg = {
        'bottleneck_width': 16,
        'cardinality': 4,
        'num_block': 3,
    }
    return ResNeXt(cfg, **kwargs)

def ResNext29_8x64d(**kwargs):
    cfg = {
        'bottleneck_width': 64,
        'cardinality': 8,
        'num_block': 3,
    }
    return ResNeXt(cfg, **kwargs)

def ResNext29_16x64d(**kwargs):
    cfg = {
        'bottleneck_width': 64,
        'cardinality': 16,
        'num_block': 3,
    }
    return ResNeXt(cfg, **kwargs)


def ResNext29_32x4d(**kwargs):
    cfg = {
        'bottleneck_width': 4,
        'cardinality': 32,
        'num_block': 3,
    }
    return ResNeXt(cfg, **kwargs)