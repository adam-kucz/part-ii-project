from pathlib import Path
from typing import List, Union

import tensorflow as tf
import tensorflow.keras.metrics as metr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

from ..abstract import DataReader
from ..modules.model_trainer import ModelTrainer
from ..modules.probability_output import ProbabilityOutput

__all__ = ['StandardStandalone', 'CategoricalStandalone']

SomeLayers = Union[Layer, List[Layer]]


class StandardStandalone(ModelTrainer):
    def __init__(self, name: str, dataset_producer: DataReader,
                 inputs: SomeLayers, outputs: SomeLayers, core: Model,
                 loss, metrics: List, out_dir: Path,
                 run_name: str = 'default', monitor: str = 'val_loss',
                 optimizer=tf.keras.optimizers.Adam()):
        super().__init__(name, dataset_producer, Model(inputs, outputs), core,
                         out_dir, run_name, monitor=monitor)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.summary()


class CategoricalStandalone(StandardStandalone):
    def __init__(self, name: str, dataset_producer: DataReader, class_num: int,
                 inputs: SomeLayers, outputs: SomeLayers, core: Model,
                 out_dir: Path, run_name: str = 'default', metrics: List = [],
                 optimizer=tf.keras.optimizers.Adam()):
        topk = metr.sparse_top_k_categorical_accuracy
        metrics = metrics + [metr.SparseCategoricalAccuracy(),
                             topk]
        probabilities = ProbabilityOutput(class_num)(outputs)
        super().__init__(name, dataset_producer, inputs, probabilities, core,
                         tf.keras.losses.CategoricalCrossentropy(),
                         metrics, out_dir, run_name,
                         monitor='val_sparse_categorical_accuracy',
                         optimizer=optimizer)
