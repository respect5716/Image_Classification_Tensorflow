import os
import tensorflow as tf
import tensorflow_addons as tfa

from utils import Dataloader
from models import vgg

def create_loader():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    train_loader = Dataloader(x_train, y_train, 'train', 256)
    test_loader = Dataloader(x_test, y_test, 'test', 256)
    return train_loader, test_loader

def create_model(model_name):
    model_dict = {
        'vgg': vgg.create_model
    }

    return model_dict[model_name]()


def main():
    train_loader, test_loader = create_loader()
    model = create_model('vgg')
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['acc'],
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.01, 1000, 0.95)
        )
    )
    print(model.summary())

    model.fit(
        train_loader,
        epochs = 10
    )

    model.evaluate(test_loader)

if __name__ == '__main__':
    main()