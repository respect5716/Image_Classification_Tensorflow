import tensorflow as tf

def residual_block(x, filters, strides):
    if strides != 1 or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides, use_bias=False)(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 3, strides, 'same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    w = tf.keras.layers.AvgPool2D(x.shape[1])(x)
    w = tf.keras.layers.Dense(filters // 16, activation='relu')(w)
    w = tf.keras.layers.Dense(filters, activation='sigmoid')(w)

    x *= w
    x += shortcut
    x = tf.keras.layers.ReLU()(x)

    return x

def residual_blocks(x, filters, num_block, downsample):
    if downsample:
        x = residual_block(x, filters, 2)
    else:
        x = residual_block(x, filters, 1)

    for _ in range(num_block-1):
        x = residual_block(x, filters, 1)
    return x

def create_senet(num_block):
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = residual_blocks(x, 64, num_block[0], False)
    x = residual_blocks(x, 128, num_block[1], True)
    x = residual_blocks(x, 256, num_block[2], True)
    x = residual_blocks(x, 512, num_block[3], True)
    
    x = tf.keras.layers.AvgPool2D(4)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def SENet18():
    return create_senet([2, 2, 2, 2])
