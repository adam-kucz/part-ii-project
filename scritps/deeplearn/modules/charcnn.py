from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Convolution1D, Dense, Flatten, Input, ReLU
from tensorflow.keras.models import Model

from ..data_ops.data_transformers import StrEnc

__all__ = ['CharCNN']


class CharCNN(Model):
    def __init__(self, params: Mapping[str, Any]):
        identifier = Input(shape=(), dtype=tf.string)
        tensor = StrEnc(params['identifier_length'])(identifier)
        for conv_params in params['convolutional']:
            tensor = Convolution1D(filters=conv_params['filters'],
                                   kernel_size=conv_params['kernel_size'],
                                   padding='valid',
                                   use_bias=False)(tensor)
            tensor = ReLU()(tensor)
        tensor = Flatten()(tensor)
        for dense_params in params['dense']:
            tensor = Dense(units=dense_params['units'])(tensor)
            tensor = ReLU()(tensor)
        super().__init__(identifier, tensor)
