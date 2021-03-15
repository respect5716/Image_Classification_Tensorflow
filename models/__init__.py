import tensorflow as tf

from .vgg import *
from .resnet import *
from .resnext import *
from .preact_resnet import *

from .densenet import *
from .dla import *
from .dpn import *

def create_model(model_name, initializer, weight_decay):
    model_dict = {
        'vgg19': VGG19,
        'resnet110': ResNet110,
        'preact_resnet110': PreactResNet110,
        'resnext29_2x64d': ResNext29_2x64d,
        'dense': DenseNetCifar,
        'dla': DLA,
        'dpn92': DPN92,
    }

    regularizer = tf.keras.regularizers.l2(weight_decay)
    kwargs = {
        'kernel_initializer': initializer,
        'kernel_regularizer': regularizer,
    }
    model = model_dict[model_name](**kwargs)
    print(model.summary())
    return model