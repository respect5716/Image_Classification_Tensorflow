"""
SENet in tensorflow 2

Jie Hu, Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507
"""

import tensorflow as tf


def residual_block(x, filters, strides, **kwargs):
    if strides != 1 or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides, use_bias=False, **kwargs)(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 3, strides, 'same', use_bias=False, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', use_bias=False, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    w = tf.keras.layers.AvgPool2D(x.shape[1])(x)
    w = tf.keras.layers.Dense(filters // 16, activation='relu')(w)
    w = tf.keras.layers.Dense(filters, activation='sigmoid')(w)

    x *= w
    x += shortcut
    x = tf.keras.layers.ReLU()(x)

    return x

def residual_stack(x, filters, num_block, downsample, block_id, **kwargs):
    if downsample:
        x = residual_block(x, filters, 2, **kwargs)
    else:
        x = residual_block(x, filters, 1, **kwargs)

    for _ in range(num_block-1):
        x = residual_block(x, filters, 1, **kwargs)
    return x

def SENet(cfg, **kwargs):
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', use_bias=False, **kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = residual_stack(x, 64, cfg['num_block'][0], False, 1, **kwargs)
    x = residual_stack(x, 128, cfg['num_block'][1], True, 2, **kwargs)
    x = residual_stack(x, 256, cfg['num_block'][2], True, 3, **kwargs)
    x = residual_stack(x, 512, cfg['num_block'][3], True, 4, **kwargs)
    
    x = tf.keras.layers.AvgPool2D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def SENet18(**kwargs):
    cfg = {
        'num_block': [2, 2, 2, 2]
    }
    return SENet(cfg, **kwargs)