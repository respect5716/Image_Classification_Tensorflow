"""
ResNet in tensorflow 2

Kaiming He, Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
"""
import tensorflow as tf

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, **kwargs):
        super(Bottleneck, self).__init__()
        self.filters = filters
        self.strides = strides
        self.expanded_filter = 4 * filters
        self.kwargs = kwargs

    def build(self, input_shape):
        if self.strides > 1 or input_shape[-1] != self.expanded_filter:
            self.shortcut = tf.keras.layers.Conv2D(self.expanded_filter, 1, self.strides, use_bias=False, **self.kwargs)
        else:
            self.shortcut = tf.identity

        self.conv1 = tf.keras.layers.Conv2D(self.filters, 1, 1, 'same', use_bias=False, **self.conv_kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
    
        self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, self.strides, 'same', use_bias=False, **self.kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(self.expanded_filter, 1, 1, 'same', use_bias=False, **self.kwargs)
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


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, num_block, **kwargs):
        super(ResBlock, self).__init__()
        strides = [strides] + [1] * (num_block - 1)
        self.blocks = [Bottleneck(filters, s, **kwargs) for s in strides]
    
    def call(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def ResNet(cfg, input_shape=(32, 32, 3), output_shape=10, **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', use_bias=False, **kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = ResBlock(64, 1, cfg['num_block'][0], **kwargs)(x)
    x = ResBlock(128, 2, cfg['num_block'][1], **kwargs)(x)
    x = ResBlock(256, 2, cfg['num_block'][2], **kwargs)(x)
    x = ResBlock(512, 2, cfg['num_block'][3], **kwargs)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax', **kwargs)(x)
    return tf.keras.Model(inputs, outputs)


def ResNet50(**kwargs):
    cfg = {
        'num_block': [3, 4, 6, 3]
    }
    return ResNet(cfg, **kwargs)

def ResNet101(**kwargs):
    cfg = {
        'num_block': [3, 4, 23, 3]
    }
    return ResNet(cfg, **kwargs)

def ResNet152(**kwargs):
    cfg = {
        'num_block': [3, 8, 36, 3]
    }
    return ResNet(cfg, **kwargs)