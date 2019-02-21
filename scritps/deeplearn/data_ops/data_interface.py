from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

from ..abstract.modules import DataInterface, DataMode, DataReader

__all__ = ['csv_reader', 'CsvReader',
           'pair_interface', 'PairInterface',
           'n_strings_interface', 'NStringsInterface']


def csv_reader(input_shape, label_shape, batch_size: int,
               path: Path, mode: DataMode = DataMode.TRAIN):
    return CsvReader(input_shape, label_shape, batch_size)(path, mode)


class CsvReader(DataReader):
    def __init__(self, input_shape, label_shape, batch_size: int) -> None:
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.label_shape = label_shape

    def __call__(self, path: Path, in_mode: Optional[DataMode] = None)\
            -> tf.data.Iterator:
        mode: DataMode = in_mode or DataMode.TRAIN
        shape: Tuple[tf.DType, ...] = ()
        if mode & DataMode.INPUTS:
            shape += self.input_shape
        if mode & DataMode.LABELS:
            shape += self.label_shape
        dataset = tf.data.experimental.CsvDataset(str(path),
                                                  shape,
                                                  header=True)
        if mode & DataMode.TRAIN:
            dataset = dataset.shuffle(1000)
        if mode & DataMode.BATCH:
            dataset = dataset.batch(self.batch_size)

        if mode & DataMode.ONEPASS:
            return dataset.make_one_shot_iterator()
        return dataset.make_initializable_iterator()


def pair_interface(handle):
    return PairInterface()(handle)


class PairInterface(DataInterface):
    def __call__(self, handle):
        iterator = tf.data.Iterator.from_string_handle(
            handle, (tf.string, tf.string), ([None], [None]))
        return iterator.get_next()

    @property
    def params(self):
        return {}


def n_strings_interface(n, handle):
    return NStringsInterface(n)(handle)


class NStringsInterface(DataInterface):
    def __init__(self, n):
        self.n = n

    def __call__(self, handle):
        iterator = tf.data.Iterator.from_string_handle(
            handle, (tf.string,) * self.n, ((None,),) * self.n)
        return iterator.get_next()

    @property
    def params(self):
        return {'nstrings': self.n}
