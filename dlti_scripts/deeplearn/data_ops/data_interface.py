from abc import abstractmethod
import math
from pathlib import Path
from typing import Tuple, List, Iterable, Optional

from funcy import (lmap, lchunks, repeat, all, isa, partial, lzip, map)
import numpy as np
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
        # print(f"Transformed dataset: {dataset}")
        return dataset


class CsvReader(DataReader):
    @abstractmethod
    def _read_dataset(self, path: Path, mode: DataMode)\
            -> Tuple[tf.data.Dataset, int]:
        pass

    @abstractmethod
    def _batch(self, dataset: tf.data.Dataset,
               dataset_size: int, mode: DataMode)\
               -> Tuple[tf.data.Dataset, int]:  # noqa: E127
        pass

    def __call__(self, path: Path, mode: DataMode = DataMode.TRAIN)\
            -> SizedDataset:
        dataset, size = self._read_dataset(path, mode)

        if mode & DataMode.SHUFFLE:
            dataset = dataset.shuffle(1000)
        if mode & DataMode.BATCH:
            dataset, size = self._batch(dataset, size, mode)

        if mode & DataMode.ONEPASS:
            return SizedDataset(dataset, size)
        return SizedDataset(dataset.repeat(), size)


class RegularCsvReader(CsvReader):
    def __init__(self, input_shape, label_shape, batch_size) -> None:
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.batch_size = batch_size

    def _read_dataset(self, path: Path, mode: DataMode)\
            -> Tuple[tf.data.Dataset, int]:
        size = len(csv_read(path))
        shape: Tuple[tf.DType, ...] = ()
        if mode & DataMode.INPUTS:
            shape += self.input_shape
        if mode & DataMode.LABELS:
            shape += self.label_shape
        dataset = tf.data.experimental.CsvDataset(str(path), shape)
        if mode & DataMode.LABELS:
            if mode & DataMode.INPUTS:
                return dataset.map(lambda *x: (x[:-1], x[-1])), size
            return dataset.map(lambda *y: ('', y)), size
        elif mode & DataMode.INPUTS:
            return dataset.map(lambda *x: (x, '')), size
        raise ValueError("Invalid data mode, "
                         "must include at least one of INPUTS or LABLES", mode)

    def _batch(self, dataset: tf.data.Dataset,
               dataset_size: int, mode: DataMode)\
               -> Tuple[tf.data.Dataset, int]:  # noqa: E127
        return (dataset.batch(self.batch_size),
                math.ceil(dataset_size / self.batch_size))


class OccurenceCsvReader(CsvReader):
    def __init__(self, ctx_size: int, batch_size: int) -> None:
        self.ctx_size = ctx_size
        self.batch_size = batch_size

    def _read_dataset(self, path: Path, mode: DataMode)\
            -> Tuple[tf.data.Dataset, int]:
        types: Tuple = ()
        shapes: Tuple = ()
        if mode & DataMode.INPUTS:
            ctx_len = self.ctx_size * 2 + 1
            types += (tf.string, tf.int8),
            shapes += ((None, None, ctx_len), (None, None)),
        types += tf.string,
        shapes += (None,),
        sequence = OccurenceCsvSequence(
            self.ctx_size, path, self.batch_size, mode)
        # print(f"Constructed sequence: {sequence}, "
        #       f"isiterable: {isinstance(sequence, Iterable)}")
        return (tf.data.Dataset.from_generator(lambda: sequence,
                                               output_types=types,
                                               output_shapes=shapes),
                len(sequence))

    def _batch(self, dataset: tf.data.Dataset,
               dataset_size: int, mode: DataMode)\
               -> Tuple[tf.data.Dataset, int]:  # noqa: E127
        return dataset, dataset_size


Context = Iterable[str]
Mask = Iterable[int]
InputBatch = Tuple[Iterable[Iterable[Context]], Iterable[Mask]]


class OccurenceCsvSequence(tf.keras.utils.Sequence):
    batch_size: int
    empty_ctx: Context
    batches_x = List[InputBatch]
    batches_y = List[Iterable[str]]

    def __init__(self, ctx_size: int, path: Path,
                 batch_size: Optional[int] = None,
                 mode: DataMode = DataMode.TRAIN) -> None:
        # print(f"Sequence for context size: {ctx_size}")
        self.ctx_len = ctx_len = 2 * ctx_size + 1
        assert batch_size is not None or not mode & DataMode.BATCH
        self.batch_size = batch_size
        raw = csv_read(path)
        if not mode & DataMode.INPUTS:
            if mode & DataMode.LABELS:
                # TODO: handle properly
                if mode & DataMode.BATCH:
                    data = lchunks(self.batch_size, raw)
                else:
                    data = raw
                self.batches = lmap(np.array, data)
                return
            raise ValueError(
                "Invalid data mode, "
                "must include at least one of INPUTS or LABLES", mode)
        batches: List[List[str]] = lchunks(self.batch_size, raw)
        inputs: Iterable[Iterable[str]]
        inputs = (batches if not mode & DataMode.LABELS
                  else ((row[:-1] for row in batch) for batch in batches))
        ctxs: Iterable[Iterable[List[Context]]]
        ctxs = ((lchunks(ctx_len, row) for row in batch) for batch in inputs)
        self.empty_ctx = [""] * ctx_len
        input_batches: InputBatch = map(self.zero_pad, ctxs)
        if mode & DataMode.LABELS:
            labels = (np.array(lmap(-1, batch)) for batch in batches)
        else:
            labels = (np.array(lmap(lambda _: '', batch))
                      for batch in batches)
        self.batches = lzip(input_batches, labels)
        for (ctxs, masks), labels in self.batches:
            lens = (len(ctxs), len(masks), len(labels))
            assert lens[0] <= self.batch_size, (ctxs, lens[0])
            assert lens[1] <= self.batch_size, (masks, lens[1])
            assert lens[2] <= self.batch_size, (labels, lens[2])
            assert lens[0] == lens[1] == lens[2], lens
        # print(f"Last batch has length {len(self.batches[-1][0])}")

    def zero_pad(self, batch: Iterable[List[Context]]) -> InputBatch:
        batch = list(batch)
        maxlen = max(map(len, batch))

        def pad_record(record: List[Context]) -> Tuple[List[Context], Mask]:
            length: int = len(record)
            padwith = (lambda s, r: r + list(repeat(s, maxlen - length)))
            return (padwith(self.empty_ctx, record), padwith(0, [1] * length))
        padded = lmap(pad_record, batch)
        for record in padded:
            ctxs, mask = record
            for ctx in ctxs:
                assert len(ctx) == self.ctx_len, (ctx, self.ctx_len)
            assert len(ctxs) == len(mask) == maxlen, record
            assert all(partial(all, isa(str)), ctxs), ctxs
            assert all(isa(int), mask), mask
        return (np.array(lmap(0, padded)), np.array(lmap(1, padded)))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx: int):
        # print(f"Getting element {idx} with label:\n{self.batches[idx][1]}\n")
        return self.batches[idx]
