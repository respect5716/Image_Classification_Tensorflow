import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images = (images / 127.5) - 1
    return images, labels

def augment(images, labels):
    images = tf.pad(images, [[4, 4], [4, 4], [0,0]])
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_crop(images, (32, 32, 3))
    return images, labels

def create_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.map(augment).map(normalize).batch(batch_size).shuffle(1000)

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.map(normalize).batch(batch_size)
    
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.map(normalize).batch(batch_size)
    return train_data, val_data, test_data