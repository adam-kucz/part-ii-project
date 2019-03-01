from abc import ABC, abstractmethod
from enum import auto, Flag, unique
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional,
                    Tuple, TypeVar, Union)

import tensorflow as tf

__all__ = ['Collection', 'Collections', 'DataInterface', 'DataMode',
           'DataReader', 'FullNet', 'OutputNet', 'Some', 'Tensors']

T = TypeVar('T')  # pylint: disable=invalid-name
Some = Union[T, Tuple[T, ...], List[T]]
Tensors = Some[tf.Tensor]
Shape = Tuple[Optional[int], ...]
Layer = tf.keras.layers.Layer
Collection = Union[str, Iterable[str]]
Collections = Optional[Union[Collection, Dict[str, Collection]]]


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


class Module(Layer, Parametrized):
    def add_sublayer(self, sublayer: Layer) -> None:
        self._trainable_weights.extend(sublayer.trainable_weights)
        self._non_trainable_weights.extend(sublayer.trainable_weights)

    @abstractmethod
    def compute_output_shape(self, input_shape: Shape) -> Shape:
        pass

    @abstractmethod
    def build(self, input_shape: Shape) -> None:
        pass

    @abstractmethod
    def call(self, inputs: Tensors) -> Tensors:
        pass


class LinearModule(Module):
    def __init__(self) -> None:
        self._sublayers: List[Layer] = []

    def add_sublayer(self, sublayer: Layer) -> None:
        super().add_sublayer(sublayer)
        self._sublayers.append(sublayer)

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        shape = input_shape
        for layer in self._sublayers:  # type: Layer
            shape = layer.compute_output_shape(shape)
        return shape

    def build(self, input_shape: Shape) -> None:
        shape = input_shape
        for layer in self._sublayers:  # type: Layer
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        super().build()

    def call(self, inputs: Tensors) -> Tensors:
        tensors = inputs
        for layer in self._sublayers:  # type: Layer
            tensors = layer(tensors)
        return tensors


class DataInterface(Module):
    """Interface with the dataset"""
    @abstractmethod
    def __call__(self, handle: tf.Tensor) -> Tensors:
        pass

class OutputNet(Parametrized):
    """
    Final layer of the network

    Responsible for defining outputs, loss, training operation and metrics
    """
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
