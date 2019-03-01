from typing import Any, Callable, Mapping

import tensorflow as tf

from ..abstract.modules import LinearModule, Tensors

__all__ = ['charcnn', 'CharCNN']


def charcnn(one_hot_chars: tf.Tensor,
            params: Mapping[str, Any],
            log: Callable[[str], None] = lambda _: None) -> tf.Tensor:
    return CharCNN(params, log)((one_hot_chars,))


class CharCNN(LinearModule):
    def __init__(self,
                 params: Mapping[str, Any],
                 log: Callable[[str], None] = lambda _: None,
                 separate_scopes: bool = True):
        self._params = params
        self._init_conv(params['convolutional'])
        self._init_dense(params['dense'])
        # self.log = log
        self.separate_scopes = separate_scopes

    def _init_convs(self, params) -> None:
        for layer_params in params:
            self.add_sublayer(tf.keras.layers.Conv1D(
                filters=layer_params['filters'],
                kernel_size=layer_params['kernel_size'],
                padding='valid', use_bias=False, activation=tf.nn.relu))

    def _init_dense(self, params) -> tf.Tensor:
        for layer_params in params:
            self.add_sublayer(tf.keras.layers.Dense(
                units=layer_params['units'], activation=tf.nn.relu))

    def __call__(self, inputs: Tensors) -> tf.Tensor:
        return super().__call__(inputs if isinstance(inputs, tf.Tensor)
                                else inputs[0])

    @property
    def params(self):
        return self._params
