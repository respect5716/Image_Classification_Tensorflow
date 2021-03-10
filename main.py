import os
import wandb
import tensorflow as tf
import tensorflow_addons as tfa

from data import create_dataset
from models import create_model
from optimizers import create_optimizer, CosineDecay

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vgg11')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--initializer', type=str, default='he_uniform')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch_size', type=int, default=200)
args = parser.parse_args()
CONFIG = vars(args)

def prepare():
    wandb.init(
        project = 'cifar10',
        config = CONFIG
    )
    
    model = create_model(CONFIG['model'], CONFIG['initializer'], CONFIG['weight_decay'])
    optimizer = create_optimizer(CONFIG['optimizer'])
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['acc'],
        optimizer = optimizer
    )
    return model

def train(model, train_data, val_data):
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(CosineDecay(CONFIG['epoch_size'], CONFIG['lr'])),
        wandb.keras.WandbCallback(monitor='val_acc'),
    ]

    history = model.fit(
        train_data,
        epochs = CONFIG['epoch_size'],
        validation_data = val_data,
        callbacks = [callbacks]
    )

def test(model, test_data):
    test_loss, test_acc = model.evaluate(test_data)
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc
    })


def main():
    train_data, val_data, test_data = create_dataset(CONFIG['batch_size'])
    model = prepare()
    train(model, train_data, val_data)
    test(model, test_data)


if __name__ == '__main__':
    main()