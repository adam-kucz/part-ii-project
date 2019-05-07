from pathlib import Path
from typing import Mapping, Counter, List, Iterable, DefaultDict, Callable

from funcy import lmap, concat, group_by, map, rpartial, identity
import numpy as np
import tensorflow as tf
from tensorflow.contrib.lookup import MutableHashTable
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
import tensorflow.keras.metrics as metr

from .standard import StandardStandalone
from ..data_ops.data_interface import CsvReader, LabelTransformingReader
from ..data_ops.data_transformers import OneHot, CategoricalIndex
from ..util import csv_read
from ..analysis.util import RecordMode, read_dataset


class Baseline(StandardStandalone):
    typs: Counter[int]
    records: Mapping[str, Counter[int]]
    table: MutableHashTable
    typ_to_index: Callable[[str], str]
    index_to_typ: Callable[[int], str]

    def __init__(self, vocab_path: Path, train_path: Path, batch_size: int,
                 out_dir: Path, run_name: str = 'default', **kwargs):
        self.typs = Counter()
        self.records = DefaultDict(Counter)
        vocab: List[str] = lmap(0, csv_read(vocab_path))
        self.typ_to_index\
            = rpartial({t: i for i, t in enumerate(vocab)}.get, len(vocab))
        self.index_to_typ = dict(enumerate(concat(vocab, '_'))).__getitem__
        index: CategoricalIndex = CategoricalIndex(vocab, unk=True)
        dataset_reader = CsvReader((tf.string,), (tf.string,), batch_size)
        dataset_reader = LabelTransformingReader(index, dataset_reader)
        identifier = Input(shape=(), dtype=tf.string)
        self.table = self._init_table(train_path)
        tensor = Lambda(self.table.lookup, output_shape=identity)(identifier)
        tensor = OneHot(len(vocab) + 1)(tensor)
        topk = metr.sparse_top_k_categorical_accuracy
        metrics = [metr.SparseCategoricalAccuracy(), topk]
        super().__init__('baseline', dataset_reader,
                         identifier, outputs=tensor, core=self.table,
                         loss=tf.keras.losses.CategoricalCrossentropy(),
                         metrics=metrics, out_dir=out_dir, run_name=run_name)

    def _init_table(self, train_path):
        trainset = read_dataset(RecordMode.IDENTIFIER, 0, train_path)
        self.typs.update(self.typ_to_index(r.label) for r in trainset)
        for identifier, vals in group_by(0, trainset).items():
            self.records[identifier].update(
                self.typ_to_index(r.label) for r in vals)
        table = MutableHashTable(key_dtype=tf.string,
                                 value_dtype=tf.int32,
                                 default_value=self.typs.most_common(1)[0][0])
        K.get_session().run(table.insert(
            keys=list(self.records),
            values=lmap(lambda c: c.most_common(1)[0][0],
                        self.records.values())))
        return table

    def full_predictions(self, valpath: Path, verbose=1)\
            -> Iterable[Iterable]:
        raw_examples = super().full_predictions(valpath, verbose)

        def process_probs(probs):
            indices = np.argsort(probs)[::-1]
            return zip(map(self.index_to_typ, indices), probs[indices])
        return map(process_probs, raw_examples)
