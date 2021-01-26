"""
VGG in tensorflow 2
with Batch normalization

Karen Simonyan, Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556v6
"""

import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, 3, 1, 'same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    
    def call(self, x):
        return self.relu(self.bn(self.conv(x)))


def VGG(cfg, input_shape=(32, 32, 3), output_shape=10):
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs

    for layer in cfg['layers']:
        if layer == 'Pool':
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = ConvBlock(layer)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def VGG11():
    cfg = {
        'layers': [64, 'Pool', 128, 'Pool', 256, 256, 'Pool', 512, 512, 'Pool', 512, 512, 'Pool']
    }
    return VGG(cfg)

def VGG13():
    cfg = {
        'layers': [64, 64, 'Pool', 128, 128, 'Pool', 256, 256, 'Pool', 512, 512, 'Pool', 512, 512, 'Pool']
    } 
    return VGG(cfg)

def VGG16():
    cfg = {
        'layers': [64, 64, 'Pool', 128, 128, 'Pool', 256, 256, 256, 'Pool', 512, 512, 512, 'Pool', 512, 512, 512, 'Pool']
    }
    return VGG(cfg)

def VGG19():
    cfg = {
        'layers': [64, 64, 'Pool', 128, 128, 'Pool', 256, 256, 256, 256, 'Pool', 512, 512, 512, 512, 'Pool', 512, 512, 512, 512, 'Pool']
    }
    return VGG(cfg)

if __name__ == '__main__':
    model = VGG11()
    print(model.summary())