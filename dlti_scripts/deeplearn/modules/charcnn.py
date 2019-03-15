from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import (
    Convolution1D, Dense, Flatten, InputLayer, ReLU)
from tensorflow.keras.models import Sequential

from ..data_ops.data_transformers import StringEncoder

__all__ = ['CharCNN']


class CharCNN(Sequential):
    def __init__(self, params: Mapping[str, Any]):
        super().__init__()
        self.add(InputLayer(input_shape=(), dtype=tf.string))
        self.add(StringEncoder(params['identifier_length']))
        for conv_params in params['convolutional']:
            self.add(Convolution1D(filters=conv_params['filters'],
                                   kernel_size=conv_params['kernel_size'],
                                   padding='valid',
                                   use_bias=False))
            self.add(ReLU())
        self.add(Flatten())
        for dense_params in params['dense']:
            self.add(Dense(units=dense_params['units']))
            self.add(ReLU())
