from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.models import Model

__all__ = ['ContextNet']


class ContextNet(Model):
    def __init__(self, params: Mapping[str, Any], name: str = ''):
        ctx_len = params['ctx_len'] * 2 + 1
        tokens = [Input(shape=(), dtype=tf.string) for i in range(ctx_len)]
        charcnn_outputs = [params['token_net'](token) for token in tokens]
        tensor = (Concatenate(axis=-1)(charcnn_outputs)
                  if len(charcnn_outputs) > 1 else charcnn_outputs[0])
        for layer_params in params['aggregate']:
            tensor = Dense(**layer_params['dense'])(tensor)
            if 'dropout' in layer_params:
                tensor = Dropout(layer_params['dropout'])(tensor)
        super().__init__(inputs=tokens, outputs=tensor)
        self.contextnet_params = params

    # def get_config(self):
    #     return {'params': self.contextnet_params}
