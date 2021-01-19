import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, 3, 1, 'same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    
    def call(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides, 'same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, 1, 'same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.shortcut = tf.keras.layers.Conv2D(filters, 1, strides, use_bias=False)
    
    def call(self, x):
        res = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        x = self.relu2(x)
        return x


class Root(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Root, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, 1, 1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu =  tf.keras.layers.ReLU()
    
    def call(self, xs):
        x = tf.concat(xs, -1)
        x = self.relu(self.bn(self.conv(x)))
        return x


class Tree(tf.keras.layers.Layer):
    def __init__(self, filters, level, strides):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(filters)
            self.left_node = ResBlock(filters, strides)
            self.right_node = ResBlock(filters, 1)
        else:
            self.root = Root(filters)
            for i in reversed(range(1, level)):
                subtree = Tree(filters, level=i, strides=strides)
                self.__setattr__(f'level_{i}', subtree)
            self.prev_root = ResBlock(filters, strides)
            self.left_node = ResBlock(filters, 1)
            self.right_node = ResBlock(filters, 1)
    
    def call(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            subtree = self.__getattribute__(f'level_{i}')
            x = subtree(x)
            xs.append(x)
        
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        x = self.root(xs)
        return x


def create_dla():
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = ConvBlock(16)(inputs)
    x = ConvBlock(16)(x)
    x = ConvBlock(32)(x)

    x = Tree(64, 1, 1)(x)
    x = Tree(128, 2, 2)(x)
    x = Tree(256, 2, 2)(x)
    x = Tree(512, 1, 2)(x)

    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

def DLA():
    return create_dla()
