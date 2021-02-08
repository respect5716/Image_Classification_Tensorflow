import os
import wandb
import tensorflow as tf
import tensorflow_addons as tfa

from utils import Dataloader
from models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vgg11')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch_size', type=int, default=200)
args = parser.parse_args()

def create_loader(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    train_loader = Dataloader(x_train, y_train, 'train', batch_size)
    test_loader = Dataloader(x_test, y_test, 'test', batch_size)
    return train_loader, test_loader

def create_model(model_name):
    model_dict = {
        'vgg11': VGG11,
        'vgg13': VGG13,
        'vgg16': VGG16,
        'vgg19': VGG19,
        'dense': DenseNetCifar,
        'dla': DLA,
        'dpn26': DPN26,
        'dpn92': DPN92,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152
    }
    return model_dict[model_name]()


def main(args):
    wandb.init(
        project = 'cifar10',
        config = vars(args)
    )

    train_loader, test_loader = create_loader(args.batch_size)
    model = create_model(args.model)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 1000, 0.95)
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['acc'],
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    )
    print(model.summary())

    model.fit(
        train_loader,
        epochs = args.epoch_size,
        validation_data = test_loader,
        callbacks = [wandb.keras.WandbCallback()]
    )

    model.evaluate(test_loader)

if __name__ == '__main__':
    main(args)