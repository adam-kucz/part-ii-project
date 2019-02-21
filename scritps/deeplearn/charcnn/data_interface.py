from pathlib import Path
from typing import Tuple

import tensorflow as tf

from ..abstract.modules import DataInterface, DataMode, DataReader

__all__ = ['csv_reader', 'CsvReader', 'pair_interface', 'PairInterface']


def csv_reader(batch_size: int, path: Path, mode: DataMode = DataMode.TRAIN):
    return CsvReader(batch_size)(path, mode)


class CsvReader(DataReader):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __call__(self,
                 path: Path,
                 mode: DataMode = DataMode.TRAIN) -> tf.data.Iterator:
        shape: Tuple[tf.DType, ...] = ()
        if mode & DataMode.INPUTS:
            shape += (tf.string,)
        if mode & DataMode.LABELS:
            shape += (tf.string,)
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
