from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, InputLayer, Lambda
from tensorflow.keras.models import Sequential

__all__ = ['OccurenceNet']


class OccurenceNet(Sequential):
    def __init__(self, params: Mapping[str, Any]):
        super().__init__()
        ctx_len = params['ctx_len'] * 2 + 1
        self.add(InputLayer(shape=(None, ctx_len), dtype=tf.string))
        self.add(Lambda(lambda elems: tf.map_fn(params['context_net'], elems)))
        self.add(GlobalAveragePooling1D())
