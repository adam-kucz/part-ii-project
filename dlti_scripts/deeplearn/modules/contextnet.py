from typing import Any, Callable, Mapping, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Flatten, Input, ReLU
from tensorflow.keras.models import Model

__all__ = ['ContextNet']


class ContextNet(Model):
    def __init__(self, params: Mapping[str, Any]):
        ctx_len = params['ctx_len']
        tokens = tuple(Input(shape=(), dtype=tf.string)
                       for i in range(ctx_len))
        charcnn_outputs = tuple(params['token_net'](token) for token in tokens)
        tensor = Flatten(0)(charcnn_outputs)
        for layer_params in params['aggregate']:
            tensor = Dense(units=layer_params['units'])(tensor)
            tensor = ReLU()(tensor)
        super().__init__(tokens, tensor)
