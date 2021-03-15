import tensorflow as tf

from .vgg import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .densenet import *
from .dpn import *

from .dla import *
from .senet import *

def create_model(model_name, initializer, weight_decay):
    model_dict = {
        'vgg': VGG11,
        'resnet': ResNet56,
        'preact_resnet': PreactResNet56,
        'resnext': ResNext29_4x16d,
        'densenet': DenseNet35,
        'dpn': DPN26,
        'senet': SENet18,
        'dla': DLA,
    }

    regularizer = tf.keras.regularizers.l2(weight_decay)
    kwargs = {
        'kernel_initializer': initializer,
        'kernel_regularizer': regularizer,
    }
    model = model_dict[model_name](**kwargs)
    print(model.summary())
    return model