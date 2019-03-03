from itertools import chain
from typing import Any, Callable, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Convolutional1D, Dense
from tensorflow.keras.models import Sequential

from ..abstract.modules import Parametrized

__all__ = ['charcnn', 'CharCNN']


def charcnn(one_hot_chars: tf.Tensor, params: Mapping[str, Any]) -> tf.Tensor:
    """Functional interface to CharCNN"""
    return CharCNN(params)((one_hot_chars,))


class CharCNN(Sequential, Parametrized):
    def __init__(self, params: Mapping[str, Any]):
        self._params = params
        for conv_params in params['convolutional']:
            self.add(Convolutional1D(filters=conv_params['filters'],
                                     kernel_size=conv_params['kernel_size'],
                                     padding='valid',
                                     use_bias=False,
                                     activation=tf.nn.relu))
        for dense_params in params['dense']:
            self.add(Dense(units=dense_params['units'], activation=tf.nn.relu))

    # def __call__(self, inputs: Tensors) -> tf.Tensor:
    #     return super().__call__(inputs if isinstance(inputs, tf.Tensor)
    #                             else inputs[0])

    @property
    def params(self):
        return self._params
