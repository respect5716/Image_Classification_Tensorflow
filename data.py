import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, x, y, mode, batch_size, transform=None):
        self.x = x
        self.y = y
        self.mode = mode
        self.batch_size = batch_size
        self.transform = iaa.Sequential([
            iaa.PadToFixedSize(36, 36),
            iaa.CropToFixedSize(32, 32),
            iaa.HorizontalFlip(p=0.5)
        ])

        self.data_size = len(self.x)
        self.on_epoch_end()
    
    def __len__(self):
        return np.ceil(self.data_size / self.batch_size).astype(np.int32)
    
    def on_epoch_end(self):
        if self.mode == 'train':
            self.indices = np.random.permutation(self.data_size)
        else:
            self.indices = np.arange(self.data_size)
    
    def __getitem__(self, idx):
        batch_idx = self.indices[self.batch_size * idx : self.batch_size * (idx+1)]
        batch_x = self.x[batch_idx]
        batch_y = self.y[batch_idx]
        if self.mode == 'train':
            batch_x = self.transform(images=batch_x)
            batch_x = np.stack(batch_x)
        
        batch_x = (batch_x / 127.5) - 1
        batch_y = batch_y[:,0]
        return batch_x, batch_y

def create_loader(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    train_loader = Dataloader(x_train, y_train, 'train', batch_size)
    test_loader = Dataloader(x_test, y_test, 'test', batch_size)
    return train_loader, test_loader