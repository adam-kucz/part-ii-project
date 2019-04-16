import math
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..abstract import DataMode, DataReader, SizedDataset
from ..util import csv_read


class CompleteRecordReader(DataReader):
    def __init__(self, original_reader: DataReader):
        self.reader = original_reader

    def __call__(self, path: Path, mode: DataMode = DataMode.TRAIN)\
            -> SizedDataset:
        dataset = self.reader(path, mode | DataMode.INPUTS | DataMode.LABELS)
        # TODO: fix empty string hack
        if mode & DataMode.LABELS:
            if mode & DataMode.INPUTS:
                return dataset
            return dataset.map(lambda _, y: ('', y))
        return dataset.map(lambda x, _: (x, ''))


class LabelTransformingReader(DataReader):
    def __init__(self, label_transform: Layer, original_reader: DataReader):
        self.reader = original_reader
        self.label_transform = label_transform

    def __call__(self, path: Path, mode: DataMode = DataMode.TRAIN)\
            -> SizedDataset:
        dataset = self.reader(path, mode)
        if mode & DataMode.LABELS:
            dataset = dataset.map(lambda x, y: (x, self.label_transform(y)))
        return dataset


class CsvReader(DataReader):
    def __init__(self, input_shape, label_shape, batch_size) -> None:
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.batch_size = batch_size

    def __call__(self, path: Path, mode: DataMode = DataMode.TRAIN)\
            -> SizedDataset:
        shape: Tuple[tf.DType, ...] = ()
        if mode & DataMode.INPUTS:
            shape += self.input_shape
        if mode & DataMode.LABELS:
            shape += self.label_shape
        dataset = tf.data.experimental.CsvDataset(str(path), shape)
        if mode & DataMode.LABELS:
            if mode & DataMode.INPUTS:
                dataset = dataset.map(lambda *x: (x[:-1], x[-1]))
            else:
                dataset = dataset.map(lambda *y: (y,))
        elif mode & DataMode.INPUTS:
            dataset = dataset.map(lambda *x: (x,))
        else:
            raise ValueError(("Invalid data mode"
                              ", must include at least one of "
                              "INPUTS or LABLES"),
                             mode)

        if mode & DataMode.SHUFFLE:
            dataset = dataset.shuffle(1000)
        size = len(csv_read(path))
        if mode & DataMode.BATCH:
            dataset = dataset.batch(self.batch_size)
            size = math.ceil(size / self.batch_size)

        if mode & DataMode.ONEPASS:
            return SizedDataset(dataset, size)
        return SizedDataset(dataset.repeat(), size)


# TODO: write RaggedCsvReader
# priority: none, just that it would look nicer


# class OccurenceCsvReader(tf.keras.utils.Sequence):
#     batch_size: int
#     x: List[Tuple[str, List[List[str]]]]
#     y: List[Tuple[str, List[List[str]]]]

#     def __init__(self, ctx_size: int, batch_size: int, path: Path,
#                  mode: DataMode = DataMode.TRAIN) -> None:
#         y, x = lmap(lambda t: (t[0], lpartition(ctx_size, t[1:])),
#                     csv_read(path))
#         pass

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx: int) -> tf.ragged.RaggedTensor:
#         return x, y
