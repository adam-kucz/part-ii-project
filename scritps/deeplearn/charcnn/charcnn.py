from typing import Any, Callable, Mapping

import tensorflow as tf

from ..abstract.modules import CoreNet

__all__ = ['charcnn', 'CharCNN']


def charcnn(one_hot_chars: tf.Tensor,
            params: Mapping[str, Any],
            log: Callable[[str], None] = lambda _: None) -> tf.Tensor:
    return CharCNN(params, log)(one_hot_chars)


# TODO: consider inheriting from tf.layers.Layer or tf.keras.layers.Layer
class CharCNN(CoreNet):
    def __init__(self,
                 params: Mapping[str, Any],
                 log: Callable[[str], None] = lambda _: None,
                 separate_scopes: bool = True):
        self.conv_params = params['convolutional']
        self.dense_params = params['dense']
        self.log = log
        self.separate_scopes = separate_scopes

    def __call__(self, one_hot_chars: tf.Tensor) -> tf.Tensor:
        self.log("Shape of one_hot_chars: {}".format(one_hot_chars.shape))
        with tf.name_scope("conv"):
            tensor = self._convolutional(one_hot_chars)
        with tf.name_scope("dense"):
            tensor = self._dense(tf.layers.flatten(tensor))
        return tensor

    def _convolutional(self, tensor: tf.Tensor) -> tf.Tensor:
        for i, layer_params in enumerate(self.conv_params):
            layer = tf.layers.Conv1D(filters=layer_params['filters'],
                                     kernel_size=layer_params['kernel_size'],
                                     padding='valid',
                                     use_bias=False,
                                     activation=tf.nn.relu)
            if self.separate_scopes:
                with tf.name_scope('conv{}'.format(i)):
                    tensor = layer(tensor)
            else:
                tensor = layer(tensor)
            self.log("Shape of tensor after convolution {}: {}"
                     .format(i, tensor.shape))
        return tensor

    def _dense(self, tensor: tf.Tensor) -> tf.Tensor:
        for i, layer_params in enumerate(self.dense_params):
            layer = tf.layers.Dense(units=layer_params['units'],
                                    activation=tf.nn.relu)
            if self.separate_scopes:
                with tf.name_scope('dense{}'.format(i)):
                    tensor = layer(tensor)
            else:
                tensor = layer(tensor)
            self.log("Shape of tensor after dense {}: {}"
                     .format(i, tensor.shape))
        return tensor

    @property
    def params(self):
        return {'conv': self.conv_params, 'dense': self.dense_params}
