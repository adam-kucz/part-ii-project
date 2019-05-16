from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

__all__ = ['OccurenceNet']


class PartialMean(Lambda):
    def __init__(self):
        super().__init__(self.partial_mean,
                         output_shape=self.compute_output_shape)

    def partial_mean(self, inputs):
        tensor, mask = inputs
        return tf.reduce_sum(tensor, 0) / tf.math.count_nonzero(mask, 0)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0]] + input_shape[2:])


class OccurenceNet(Model):
    def __init__(self, params: Mapping[str, Any]):
        ctx_len = params['ctx_len'] * 2 + 1
        occurences = Input(shape=(None, ctx_len), dtype=tf.string)
        contextnet = params['context_net']

        def elementwise_contextnet(elems):
            return tf.map_fn(contextnet, elems)
        tensor = Lambda(elementwise_contextnet)(occurences)
        mask = Input(shape=(None, ctx_len), dtype=tf.int8)
        output = PartialMean()([tensor, mask])
        super().__init__(inputs=[occurences, mask], outputs=output)
