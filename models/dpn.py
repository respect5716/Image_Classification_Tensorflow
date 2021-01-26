"""
DPN in tensorflow 2

Yunpeng Chen, Dual Path Network
https://arxiv.org/abs/1707.01629
"""
import tensorflow as tf

class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(GroupConv2D, self).__init__()
        self.filters = filters
        self.strides = strides
        self.num_group = 32

    def build(self, input_shape):
        assert self.filters % self.num_group == 0
        group_filters = self.filters // self.num_group
        self.convs = [tf.keras.layers.Conv2D(group_filters, 3, self.strides, 'same', use_bias=False) for _ in range(self.num_group)]

    def call(self, x):
        xs = tf.split(x, self.num_group, axis=-1)
        xs = [c(x) for x, c in zip(xs, self.convs)]
        x = tf.concat(xs, axis=-1)
        return x


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters1, filters2, dense_depth, strides, is_first):
        super(Bottleneck, self).__init__()
        self.filters2 = filters2

        self.conv1 = tf.keras.layers.Conv2D(filters1, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = GroupConv2D(filters1, strides)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(filters2 + dense_depth, 1, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        if is_first:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters2 + dense_depth, 1, strides, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = tf.identity
    
    def call(self, x):
        res = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        d = self.filters2
        x = tf.concat([x[:,:,:,:d] + res[:,:,:,:d], x[:,:,:,d:], res[:,:,:,d:]], axis=-1)
        x = self.relu3(x)
        return x

def DPN(cfg, input_shape=(32, 32, 3), output_shape=10):
    filters1, filters2, = cfg['filters1'], cfg['filters2']
    num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']
    strides = [1, 2, 2, 2]

    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for f1, f2, nb, dp, st in zip(filters1, filters2, num_blocks, dense_depth, strides):
        block_strides = [st] + [1] * (nb - 1)
        for idx, bs in enumerate(block_strides):
            x = Bottleneck(f1, f2, dp, bs, idx==0)(x)
    
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

def DPN26():
    cfg = {
        'filters1': [96, 192, 384, 768],
        'filters2': [256, 512, 1024, 2048],
        'num_blocks': [2, 2, 2, 2],
        'dense_depth': [16, 32, 24, 128]
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'filters1': [96, 192, 384, 768],
        'filters2': [256, 512, 1024, 2048],
        'num_blocks': [3, 4, 20, 3],
        'dense_depth': [16, 32, 24, 128]
    }
    return DPN(cfg)