"""
PreactResNet in tensorflow 2

Kaiming He, Identity Mappings in Deep Residual Networks
https://arxiv.org/abs/1603.05027
"""

import tensorflow as tf

class PreactResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, **kwargs):
        super(PreactResBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.kwargs = kwargs
    
    def build(self, input_shape):
        if self.strides > 1:
            self.shortcut = tf.keras.layers.Conv2D(self.filters, 1, self.strides, use_bias=False, **self.kwargs)
        else:
            self.shortcut = tf.identity

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(self.filters, 3, self.strides, 'same', use_bias=False, **self.kwargs)
    
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, 1, 'same', use_bias=False, **self.kwargs)

    
    def call(self, x):
        res = self.shortcut(x)
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x += res
        return x


class PreactResStack(tf.keras.layers.Layer):
    def __init__(self, filters, strides, num_block, **kwargs):
        super(PreactResStack, self).__init__()
        strides = [strides] + [1] * (num_block - 1)
        self.blocks = [PreactResBlock(filters, s, **kwargs) for s in strides]
    
    def call(self, x):
        for b in self.blocks:
            x = b(x)
        return x

def PreactResNet(cfg, input_shape=(32, 32, 3), output_shape=10, **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(16, 3, 1, 'same', use_bias=False, **kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = PreactResStack(16, 1, cfg['num_block'], **kwargs)(x)
    x = PreactResStack(32, 2, cfg['num_block'], **kwargs)(x)
    x = PreactResStack(64, 2, cfg['num_block'], **kwargs)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax', **kwargs)(x)
    return tf.keras.Model(inputs, outputs)

def PreactResNet56(**kwargs):
    cfg = {
        'num_block': 9
    }
    return PreactResNet(cfg, **kwargs)

def PreactResNet110(**kwargs):
    cfg = {
        'num_block': 18
    }
    return PreactResNet(cfg, **kwargs)