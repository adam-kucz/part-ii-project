from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..abstract import DataMode, DataReader

__all__ = ['CsvReader']


class CsvReader(DataReader):
    def __init__(self, input_shape, label_shape,
                 label_transform: Layer, batch_size) -> None:
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.label_transform = label_transform
        self.batch_size = batch_size

    def __call__(self, path: Path, in_mode: Optional[DataMode] = None)\
            -> tf.data.Dataset:
        mode: DataMode = in_mode or DataMode.TRAIN
        shape: Tuple[tf.DType, ...] = ()
        if mode & DataMode.INPUTS:
            shape += self.input_shape
        if mode & DataMode.LABELS:
            shape += self.label_shape
        dataset = tf.data.experimental.CsvDataset(str(path), shape)
        if mode & DataMode.LABELS:
            if mode & DataMode.INPUTS:
                dataset = dataset.map(
                    lambda *x: (x[:-1], self.label_transform(x[-1])))
            else:
                dataset = dataset.map(lambda *x: ((), self.label_transform(x)))
        else:
            dataset = dataset.map(lambda *x: (x, ()))

        if mode & DataMode.SHUFFLE:
            dataset = dataset.shuffle(1000)
        if mode & DataMode.BATCH:
            dataset = dataset.batch(self.batch_size)

        if mode & DataMode.ONEPASS:
            return dataset
        return dataset.repeat()
