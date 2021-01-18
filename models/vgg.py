import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, 3, 1, 'same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    
    def call(self, x):
        return self.relu(self.bn(self.conv(x)))

def create_model():
    """
    VGG16 with Batch Normalization
    """
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = ConvBlock(64)(inputs)
    x = ConvBlock(64)(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = ConvBlock(128)(x)
    x = ConvBlock(128)(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = ConvBlock(256)(x)
    x = ConvBlock(256)(x)
    x = ConvBlock(256)(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = ConvBlock(512)(x)
    x = ConvBlock(512)(x)
    x = ConvBlock(512)(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = ConvBlock(512)(x)
    x = ConvBlock(512)(x)
    x = ConvBlock(512)(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)