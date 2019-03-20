from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import (
    Convolution1D, Dense, Dropout, Flatten, InputLayer, ReLU, SpatialDropout1D)
from tensorflow.keras.models import Sequential

from ..data_ops.data_transformers import StringEncoder

__all__ = ['CharCNN']


class CharCNN(Sequential):
    def __init__(self, params: Mapping[str, Any]):
        super().__init__()
        self.add(InputLayer(input_shape=(), dtype=tf.string))
        self.add(StringEncoder(params['identifier_length']))
        for layer_params in params['convolutional']:
            self.add(Convolution1D(**layer_params['conv']))
            if 'dropout' in layer_params:
                self.add(SpatialDropout1D(layer_params['dropout']))
        self.add(Flatten())
        for layer_params in params['dense']:
            self.add(Dense(**layer_params['dense']))
            if 'dropout' in layer_params:
                self.add(Dropout(layer_params['dropout']))
