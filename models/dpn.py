"""
DPN in tensorflow 2

Yunpeng Chen, Dual Path Network
https://arxiv.org/abs/1707.01629
"""
import tensorflow as tf

class Block(tf.keras.layers.Layer):
    def __init__(self, filters1, filters2, dense_depth, strides, **kwargs):
        super(Block, self).__init__()
        self.filters1 = filters1
        self.filters2 = filters2
        self.dense_depth = dense_depth
        self.output_filters = filters2 + dense_depth
        self.strides = strides
        self.kwargs = kwargs

    def build(self, input_shape):
        if self.strides > 1 or input_shape[-1] != self.output_filters:
            self.shortcut = tf.keras.layers.Conv2D(self.output_filters, 1, self.strides, use_bias=False, **self.kwargs)
        else:
            self.shortcut = tf.identity

        self.conv1 = tf.keras.layers.Conv2D(self.filters1, 1, 1, 'same', use_bias=False, **self.kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
    
        self.conv2 = tf.keras.layers.Conv2D(self.filters1, 3, self.strides, 'same', groups=8, use_bias=False, **self.kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(self.output_filters, 1, 1, 'same', use_bias=False, **self.kwargs)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
    
    def call(self, x):
        res = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        d = self.filters2
        x = tf.concat([x[:,:,:,:d] + res[:,:,:,:d], x[:,:,:,d:], res[:,:,:,d:]], axis=-1)
        x = self.relu3(x)
        return x

class Stack(tf.keras.layers.Layer):
    def __init__(self, filters1, filters2, dense_depth, strides, num_block, **kwargs):
        super(Stack, self).__init__()
        strides = [strides] + [1] * (num_block - 1)
        self.blocks = [Block(filters1, filters2, dense_depth, st, **kwargs) for st in strides]
    
    def call(self, x):
        for b in self.blocks:
            x = b(x)
        return x

def DPN(cfg, input_shape=(32, 32, 3), output_shape=10, **kwargs):
    f1, f2, dp, nb = cfg['filters1'], cfg['filters2'], cfg['dense_depth'], cfg['num_block']

    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(16, 3, 1, 'same', use_bias=False, **kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Stack(f1[0], f2[0], dp[0], 1, nb[0])(x)
    x = Stack(f1[1], f2[1], dp[1], 2, nb[1])(x)
    x = Stack(f1[2], f2[2], dp[2], 2, nb[2])(x)
    x = Stack(f1[3], f2[3], dp[3], 2, nb[3])(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def DPN32(**kwargs):
    cfg = {
        'filters1': [16, 32, 64, 128],
        'filters2': [48, 96, 192, 384],
        'dense_depth': [4, 8, 10, 20],
        'num_block': [2, 3, 3, 2],
    }
    return DPN(cfg, **kwargs)

def DPN98(**kwargs):
    cfg = {
        'filters1': [16, 32, 64, 128],
        'filters2': [48, 96, 192, 384],
        'dense_depth': [4, 8, 10, 20],
        'num_block': [2, 3, 3, 2],
    }
    return DPN(cfg, **kwargs)