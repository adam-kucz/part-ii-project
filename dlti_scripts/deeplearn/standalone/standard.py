from pathlib import Path
from typing import Iterable, List, Mapping, Union

from funcy import lmap
import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as metr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

from ..abstract import DataReader
from ..data_ops.data_interface import LabelTransformingReader
from ..data_ops.data_transformers import CategoricalIndex
from ..modules.model_trainer import ModelTrainer
from ..modules.probability_output import ProbabilityOutput
from ..util import csv_read

__all__ = ['StandardStandalone', 'CategoricalStandalone']

SomeLayers = Union[Layer, List[Layer]]


class StandardStandalone(ModelTrainer):
    def __init__(self, name: str, dataset_producer: DataReader,
                 inputs: SomeLayers, outputs: SomeLayers, core: Model,
                 loss, metrics: List, out_dir: Path,
                 run_name: str = 'default', monitor: str = 'val_loss',
                 optimizer=tf.train.AdamOptimizer()):
        super().__init__(name, dataset_producer, Model(inputs, outputs), core,
                         out_dir, run_name, monitor=monitor)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.summary()


class CategoricalStandalone(StandardStandalone):
    def __init__(self, name: str, dataset_producer: DataReader, class_num: int,
                 inputs: SomeLayers, outputs: SomeLayers, core: Model,
                 out_dir: Path, run_name: str = 'default', metrics: List = [],
                 optimizer=tf.train.AdamOptimizer()):
        topk = metr.sparse_top_k_categorical_accuracy
        metrics = metrics + [metr.SparseCategoricalAccuracy(), topk]
        probabilities = ProbabilityOutput(class_num)(outputs)
        super().__init__(name, dataset_producer, inputs, probabilities, core,
                         tf.keras.losses.CategoricalCrossentropy(),
                         metrics, out_dir, run_name,
                         monitor='val_loss',
                         optimizer=optimizer)


class VocabCategoricalStandalone(CategoricalStandalone):
    def __init__(self, name: str,
                 dataset_reader: DataReader, vocab_path: Path,
                 inputs: SomeLayers, outputs: SomeLayers, core: Model,
                 out_dir: Path, run_name: str = 'default', metrics: List = [],
                 optimizer=tf.train.AdamOptimizer()):
        vocab: List[str] = lmap(0, csv_read(vocab_path))
        self.index_to_typ: Mapping[int, str] = {i: typ
                                                for i, typ in enumerate(vocab)}
        self.index_to_typ[len(vocab)] = '_'
        index: CategoricalIndex = CategoricalIndex(vocab, unk=True)
        dataset_reader = LabelTransformingReader(index, dataset_reader)
        super().__init__(name, dataset_reader, len(vocab) + 1,
                         inputs, outputs, core,
                         out_dir, run_name, metrics, optimizer)

    def full_predictions(self, valpath: Path, verbose=1)\
            -> Iterable[Iterable]:
        raw_examples = super().full_predictions(valpath, verbose)

        def process_probs(probs):
            indices = np.argsort(probs)[::-1]
            return zip(map(lambda ind: self.index_to_typ[ind], indices),
                       probs[indices])
        return map(process_probs, raw_examples)
        # return map(lambda x, y, pred:
        #            (x, self.index_to_typ[y], self.index_to_typ[pred]),
        #            raw_examples)
