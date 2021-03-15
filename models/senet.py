"""
SENet in tensorflow 2

Jie Hu, Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507
"""

import tensorflow as tf

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides, **kwargs):
        super(Block, self).__init__()
        self.filters = filters
        self.strides = strides
        self.kwargs = kwargs

    def build(self, input_shape):
        if self.strides > 1 or input_shape[-1] != self.filters:
            self.shortcut = tf.keras.layers.Conv2D(self.filters, 1, self.strides, use_bias=False, **self.kwargs)
        else:
            self.shortcut = tf.identity

        self.conv1 = tf.keras.layers.Conv2D(self.filters, 3, self.strides, 'same', use_bias=False, **self.kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, self.strides, 'same', use_bias=False, **self.kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.pool = tf.keras.layers.GlobalAvgPool2D()
        self.dense1 = tf.keras.layers.Dense(self.filters // 16, activation='relu', **self.kwargs)
        self.dense2 = tf.keras.layers.Dense(self.filters, activation='sigmoid', **self.kwargs)
        self.relu2 = tf.keras.layers.ReLU()
    
    def call(self, x):
        res = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        w = self.dense2(self.dense1(self.pool(x)))
        x *= w
        x += res
        x = self.relu2(x)
        return x

class Stack(tf.keras.layers.Layer):
    def __init__(self, filters, strides, num_block, **kwargs):
        super(Stack, self).__init__()
        strides = [strides] + [1] * (num_block - 1)
        self.blocks = [Block(filters, st, **kwargs) for st in strides]
    
    def call(self, x):
        for b in self.blocks:
            x = b(x)
        return x

def SENet(cfg, input_shape=(32, 32, 3), output_shape=10, **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', use_bias=False, **kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = Stack(16, 1, cfg['num_block'][0], **kwargs)(x)
    x = Stack(32, 2, cfg['num_block'][1], **kwargs)(x)
    x = Stack(64, 2, cfg['num_block'][2], **kwargs)(x)
    x = Stack(128, 2, cfg['num_block'][3], **kwargs)(x)
    
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def SENet26(**kwargs):
    cfg = {
        'num_block': [3, 3, 3, 3]
    }
    return SENet(cfg, **kwargs)