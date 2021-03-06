import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def create_optimizer(optim_name):
    if optim_name == 'sgd':
        return tf.keras.optimizers.SGD(momentum=0.9)
    elif optim_name == 'adam':
        return tf.keras.optimizers.Adam()
    elif optim_name == 'radam':
        return tfa.optimizers.RectifiedAdam()

def CosineDecay(T_max, eta_max, eta_min=0):
    def cosine_decay_fn(epoch, lr=None):
        epoch = min(T_max, epoch)
        fraction = epoch / T_max
        lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(fraction * np.pi))
        return lr
    return cosine_decay_fn
