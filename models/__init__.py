import tensorflow as tf

from .vgg import *
from .densenet import *
from .dla import *
from .dpn import *
from .resnet import *

def create_model(model_name, weight_decay):
    model_dict = {
        'vgg11': VGG11,
        'vgg13': VGG13,
        'vgg16': VGG16,
        'vgg19': VGG19,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'dense': DenseNetCifar,
        'dla': DLA,
        'dpn26': DPN26,
        'dpn92': DPN92,
    }

    initializer = tf.keras.initializers.HeUniform()
    regularizer = tf.keras.regularizers.L2(weight_decay)
    model = model_dict[model_name](initializer, regularizer)
    return model