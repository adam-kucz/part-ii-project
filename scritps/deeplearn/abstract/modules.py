from abc import ABC, abstractmethod
from enum import auto, Flag, unique
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import tensorflow as tf

__all__ = ['CoreNet', 'DataInterface', 'DataMode', 'DataProcessor',
           'DataReader', 'FullNet', 'OutputNet']


Tensors = Tuple[tf.Tensor, ...]


@unique
class DataMode(Flag):
    """
    Mode for reading a dataset

    INPUTS - inputs are present
    LABELS - labels are present
    BATCH - dataset batched
    SHUFFLE - dataset will be shuffled
    ONEPASS - dataset suitable for only one pass
    TRAIN - batched, inputs and labels with shuffling
    TEST - batched, inputs and labels, no shuffling
    PREDICT - batched, inputs only, no shuffling, one pass only
    """
    INPUTS = auto()
    LABELS = auto()
    BATCH = auto()
    SHUFFLE = auto()
    ONEPASS = auto()
    TRAIN = INPUTS | LABELS | BATCH | SHUFFLE
    TEST = INPUTS | LABELS | BATCH
    PREDICT = INPUTS | BATCH | ONEPASS


class DataReader(ABC):
    @abstractmethod
    def __call__(self, path: Path,
                 mode: Optional[DataMode]) -> tf.data.Iterator:
        pass


class Parametrized(ABC):
    @property
    @abstractmethod
    def params(self) -> Mapping[str, Any]:
        pass


class DataInterface(Parametrized):
    @abstractmethod
    def __call__(self, handle: tf.Tensor) -> Tensors:
        pass


class DataProcessor(Parametrized):
    @abstractmethod
    def __call__(self, data_tensor: Tensors) -> Tuple[Tensors, Tensors]:
        pass


class CoreNet(Parametrized):
    @abstractmethod
    def __call__(self, inputs: Tensors) -> Tensors:
        pass 


class OutputNet(Parametrized):
    @abstractmethod
    def __call__(self, inputs: Tensors, labels: Tensors) -> 'OutputNet':
        pass

    @property
    @abstractmethod
    def epoch(self):
        pass

    @property
    @abstractmethod
    def increment_epoch(self):
        pass

    @property
    @abstractmethod
    def learning_rate(self):
        pass

    @property
    @abstractmethod
    def train_op(self):
        pass

    @property
    @abstractmethod
    def metric_vals(self):
        pass

    @property
    @abstractmethod
    def metric_ops(self):
        pass


class FullNet(Parametrized):
    @abstractmethod
    def __call__(self, data: Tensors) -> OutputNet:
        pass

    @property
    @abstractmethod
    def out(self) -> OutputNet:
        pass
