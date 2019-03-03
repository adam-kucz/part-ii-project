from abc import ABC, abstractmethod
from enum import auto, Flag, unique
from pathlib import Path
from typing import (Any, Dict, Iterable, Generic, List, Mapping, Optional,
                    Tuple, TypeVar, Union)

import tensorflow as tf

__all__ = ['Collection', 'Collections', 'DataMode',
           'DataReader', 'FullNet', 'OutputNet', 'Some', 'Tensors']

T = TypeVar('T')  # pylint: disable=invalid-name
S = TypeVar('S')  # pylint: disable=invalid-name
Some = Union[T, Tuple[T, ...], List[T]]
Tensors = Some[tf.Tensor]
Shape = Tuple[Optional[int], ...]
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
                 mode: Optional[DataMode]) -> tf.data.Dataset:
        pass


class Parametrized(ABC):
    @property
    @abstractmethod
    def params(self) -> Mapping[str, Any]:
        pass


class Processor(Parametrized, Generic[S, T]):
    @abstractmethod
    def __call__(self, a: S) -> T:
        pass


# notes on keras layers
#
# weights are added in build()
# losses are added in call()
# updates are added in call()
# TODO: consider if it can be replaced by tf.keras.engine.Network
class Module(tf.keras.layers.Layer, Parametrized):
    _sublayers: List[tf.keras.layers.Layer]

    def __init__(self):
        super().__init__()
        self._sublayers = []

    @property
    def sublayers(self) -> Iterable[tf.keras.layers.Layer]:
        """Adds sublayer to the module, do not call after build()"""
        return self._sublayers[:]

    def add_sublayer(self, sublayer: tf.keras.layers.Layer) -> None:
        self._sublayers.append(sublayer)

    def get_trainable_weight_dict(self) -> Dict[str, tf.Variable]:
        weights = {}
        for i, layer in enumerate(self.sublayers):
            for j, weight in enumerate(layer.trainable_weights):
                weights["layer{}_w{}".format(i, j)] = weight
        return weights

    @abstractmethod
    def compute_output_shape(self, input_shape: Shape) -> Shape:
        pass

    @abstractmethod
    def build(self, input_shape: Shape) -> None:
        for sublayer in self._sublayers:  # type: tf.keras.layers.Layer
            self._trainable_weights.extend(sublayer.trainable_weights)
            self._non_trainable_weights.extend(sublayer.trainable_weights)
        super().build(input_shape)

    @abstractmethod
    def call(self, inputs: Tensors, **kwargs) -> Tensors:
        for sublayer in self._sublayers:  # type: tf.keras.layers.Layer
            self._updates.extend(sublayer.updates)
            self._losses.extend(sublayer.losses)
        return super().call(inputs, kwargs)


# TODO: consider if it can be replaced by tf.keras.models.Sequential
class LinearModule(Module):
    def __init__(self, layers: Iterable[tf.keras.layers.Layer]) -> None:
        super().__init__()
        for layer in layers:  # type: tf.keras.layers.Layer
            self.add_sublayer(layer)

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        shape = input_shape
        for layer in self.sublayers:  # type: tf.keras.layers.Layer
            shape = layer.compute_output_shape(shape)
        return shape

    def build(self, input_shape: Shape) -> None:
        shape = input_shape
        for layer in self.sublayers:  # type: tf.keras.layers.Layer
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        super().build(input_shape)

    def call(self, inputs: Tensors, **kwargs) -> Tensors:
        super().call(inputs, **kwargs)
        tensors = inputs
        for layer in self.sublayers:  # type: tf.keras.layers.Layer
            tensors = layer(tensors)
        return tensors


class OutputNet(tf.keras.Model):
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


# TODO: consider also subclassing tf.keras.layers.Model
class FullNet(Parametrized):
    @abstractmethod
    def __call__(self, data: Tensors) -> OutputNet:
        pass

    @property
    @abstractmethod
    def out(self) -> OutputNet:
        pass
