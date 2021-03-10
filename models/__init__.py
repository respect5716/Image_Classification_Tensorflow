import tensorflow as tf

from .vgg import *
from .resnet import *
from .resnext import *

from .densenet import *
from .dla import *
from .dpn import *

def create_model(model_name, initializer, weight_decay):
    model_dict = {
        'vgg11': VGG11,
        'vgg13': VGG13,
        'vgg16': VGG16,
        'vgg19': VGG19,
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet44': ResNet44,
        'resnet56': ResNet56,
        'resnet110': ResNet110,
        'resnext29_8x64d': ResNext29_8x64d,
        'resnext29_16x64d': ResNext29_16x64d,
        'resnext29_32x4d': ResNext29_32x4d,
        'dense': DenseNetCifar,
        'dla': DLA,
        'dpn26': DPN26,
        'dpn92': DPN92,
    }

    regularizer = tf.keras.regularizers.L2(weight_decay)
    kwargs = {
        'kernel_initializer': initializer,
        'kernel_regularizer': regularizer,
    }
    model = model_dict[model_name](**kwargs)
    print(model.summary())
    return model