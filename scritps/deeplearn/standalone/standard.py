from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import tensorflow as tf
import tensorflow.keras.metrics as metr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

from ..abstract import DataReader
from ..modules.model_trainer import ModelTrainer
from ..modules.probability_output import ProbabilityOutput

__all__ = ['StandardStandalone', 'CategoricalStandalone']

SomeLayers = Union[Layer, Tuple[Layer, ...], List[Layer]]


class StandardStandalone(ModelTrainer):
    def __init__(self, name: str, dataset_producer: DataReader,
                 inputs: SomeLayers, outputs: SomeLayers,
                 loss, metrics: List,
                 out_dir: Path, run_name: str = 'default'):
        super().__init__(name, dataset_producer, Model(inputs, outputs),
                         out_dir, run_name)
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss=loss, metrics=metrics)
        self.model.summary()


class CategoricalStandalone(StandardStandalone):
    def __init__(self, name: str, dataset_producer: DataReader,
                 class_num: int, inputs: SomeLayers, outputs: SomeLayers,
                 out_dir: Path, run_name: str = 'default', metrics: List = []):
        topk = metr.sparse_top_k_categorical_accuracy
        metrics = metrics + [metr.SparseCategoricalAccuracy(),
                             topk]
        probabilities = ProbabilityOutput(class_num)(outputs)
        super().__init__(name, dataset_producer, inputs, probabilities,
                         tf.keras.losses.CategoricalCrossentropy(),
                         metrics, out_dir, run_name)
