import tensorflow as tf

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(Bottleneck, self).__init__()
        self.filters = filters
        self.strides = strides
        self.expansion = 4
    
    def build(self, input_shape):
        if self.strides > 1 or input_shape[-1] != self.filters:
            self.shortcut = tf.keras.layers.Conv2D(self.filters * self.expansion, 1, self.strides, use_bias=False)
        else:
            self.shortcut = tf.identity

        self.conv1 = tf.keras.layers.Conv2D(self.filters, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
    

        if self.strides > 1:
            self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, 2, 'same', use_bias=False)
        else:
            self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, 1, 'same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(self.filters * self.expansion, 1, use_bias=False)
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
    def __init__(self, filters, num_block, downsample):
        super(ResBlock, self).__init__()
        if downsample:
            self.blocks = [Bottleneck(filters, 1) for i in range(num_block)]
        else:
            self.blocks = [Bottleneck(filters, 2) if i==0 else Bottleneck(filters, 1) for i in range(num_block)]
    
    def call(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def create_resnet(cfg):
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = ResBlock(64, cfg[0], False)(x)
    x = ResBlock(128, cfg[1], True)(x)
    x = ResBlock(256, cfg[2], True)(x)
    x = ResBlock(512, cfg[3], True)(x)

    x = tf.keras.layers.AvgPool2D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def ResNet50():
    cfg = [3, 4, 6, 3]
    return create_resnet(cfg)

def ResNet101():
    cfg = [3, 4, 23, 3]
    return create_resnet(cfg)

def ResNet152():
    cfg = [3, 8, 36, 3]
    return create_resnet(cfg)