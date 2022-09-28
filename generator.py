import tensorflow as tf
import math


class TestDataSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]

        return (batch_x,)
