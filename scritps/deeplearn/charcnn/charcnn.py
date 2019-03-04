from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Convolution1D, Dense, ReLU
from tensorflow.keras.models import Sequential

from ..data_ops.data_transformers import StrEnc

__all__ = ['charcnn', 'CharCNN']


def charcnn(one_hot_chars: tf.Tensor, params: Mapping[str, Any]) -> tf.Tensor:
    """Functional interface to CharCNN"""
    return CharCNN(params)((one_hot_chars,))


class CharCNN(Sequential):
    def __init__(self, params: Mapping[str, Any]):
        super().__init__()
        self.add(StrEnc(params['identifier_length']))
        for conv_params in params['convolutional']:
            self.add(Convolution1D(filters=conv_params['filters'],
                                   kernel_size=conv_params['kernel_size'],
                                   padding='valid',
                                   use_bias=False))
            self.add(ReLU())
        for dense_params in params['dense']:
            self.add(Dense(units=dense_params['units']))
            self.add(ReLU())
