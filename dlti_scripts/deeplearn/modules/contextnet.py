from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, ReLU
from tensorflow.keras.models import Model

__all__ = ['ContextNet']


class ContextNet(Model):
    def __init__(self, params: Mapping[str, Any]):
        ctx_len = params['ctx_len'] * 2 + 1
        tokens = [Input(shape=(), dtype=tf.string) for i in range(ctx_len)]
        charcnn_outputs = [params['token_net'](token) for token in tokens]
        tensor = Concatenate(axis=-1)(charcnn_outputs)
        for layer_params in params['aggregate']:
            tensor = Dense(units=layer_params['units'])(tensor)
            tensor = ReLU()(tensor)
            tensor = Dropout(layer_params.get("dropout", 0))(tensor)
        super().__init__(tokens, tensor)
