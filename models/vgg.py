"""
VGG in tensorflow 2
with Batch normalization

Karen Simonyan, Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556v6
"""

import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, 3, 1, 'same', **kwargs)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
    
    def call(self, x):
        return self.relu(self.bn(self.conv(x)))


def VGG(cfg, input_shape=(32, 32, 3), output_shape=10, **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs

    for layer in cfg['layers']:
        if layer == 'POOL':
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = ConvBlock(layer, **kwargs)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax', **kwargs)(x)
    return tf.keras.Model(inputs, outputs)


def VGG11(**kwargs):
    cfg = {
        'layers': [64, 'POOL', 128, 'POOL', 256, 256, 'POOL', 512, 512, 'POOL', 512, 512, 'POOL']
    }
    return VGG(cfg, **kwargs)

def VGG13(**kwargs):
    cfg = {
        'layers': [64, 64, 'POOL', 128, 128, 'POOL', 256, 256, 'POOL', 512, 512, 'POOL', 512, 512, 'POOL']
    } 
    return VGG(cfg, **kwargs)

def VGG16(**kwargs):
    cfg = {
        'layers': [64, 64, 'POOL', 128, 128, 'POOL', 256, 256, 256, 'POOL', 512, 512, 512, 'POOL', 512, 512, 512, 'POOL']
    }
    return VGG(cfg, **kwargs)

def VGG19(**kwargs):
    cfg = {
        'layers': [64, 64, 'POOL', 128, 128, 'POOL', 256, 256, 256, 256, 'POOL', 512, 512, 512, 512, 'POOL', 512, 512, 512, 512, 'POOL']
    }
    return VGG(cfg, **kwargs)