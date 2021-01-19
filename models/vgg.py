import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, 3, 1, 'same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    
    def call(self, x):
        return self.relu(self.bn(self.conv(x)))


def create_vggnet(cfg):
    """
    VGG with batch normalization
    """
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = inputs

    for c in cfg:
        if c == 'Pool':
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = ConvBlock(c)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def VGG11():
    cfg = [64, 'Pool', 128, 'Pool', 256, 256, 'Pool', 512, 512, 'Pool', 512, 512, 'Pool']
    return create_vggnet(cfg)

def VGG13():
    cfg = [64, 64, 'Pool', 128, 128, 'Pool', 256, 256, 'Pool', 512, 512, 'Pool', 512, 512, 'Pool']
    return create_vggnet(cfg)

def VGG16():
    cfg = [64, 64, 'Pool', 128, 128, 'Pool', 256, 256, 256, 'Pool', 512, 512, 512, 'Pool', 512, 512, 512, 'Pool']
    return create_vggnet(cfg)

def VGG19():
    cfg = [64, 64, 'Pool', 128, 128, 'Pool', 256, 256, 256, 256, 'Pool', 512, 512, 512, 512, 'Pool', 512, 512, 512, 512, 'Pool']
    return create_vggnet(cfg)