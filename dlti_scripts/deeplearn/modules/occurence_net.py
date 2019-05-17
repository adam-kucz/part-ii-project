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
        weight = tf.cast(tf.math.count_nonzero(mask, 1), dtype=tf.float32)
        return tf.reduce_sum(tensor, 1) / weight[:, tf.newaxis]
        # print(f"Tensor: {tensor}, mask: {mask}, weight: {weight}")
        # print_op = tf.print("Executing, Tensor:", tf.shape(tensor),
        #                     ", mask: ", tf.shape(mask),
        #                     ", weights: ", tf.shape(weight))
        # with tf.control_dependencies([print_op]):
        #     result = tf.reduce_sum(tensor, 1) / weight[:, tf.newaxis]
        # return result

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0]] + input_shape[2:])


class OccurenceNet(Model):
    def __init__(self, params: Mapping[str, Any]):
        ctx_len = params['ctx_size'] * 2 + 1
        occurences = Input(shape=(None, ctx_len), dtype=tf.string)
        contextnet = params['context_net']

        def apply_contextnet(tensor):
            # print(f"Applying contextnet to {tensor}")
            return contextnet(tf.unstack(tensor, axis=1))

        def elementwise_contextnet(contexts):
            # print(f"Contexts: {contexts}")
            transposed = tf.transpose(contexts, perm=[1, 0, 2])
            # print(f"Transposed: {transposed}")
            applied = tf.map_fn(apply_contextnet, transposed, dtype=tf.float32)
            # print(f"Applied: {applied}")
            final = tf.transpose(applied, perm=[1, 0, 2])
            # print(f"Final: {final}")
            # print_op = tf.print("Applying ctxnet, contexts:",
            #                     tf.shape(contexts),
            #                     ", transposed: ",
            #                     tf.shape(transposed),
            #                     ", applied: ",
            #                     tf.shape(applied),
            #                     ", final: ",
            #                     tf.shape(final))
            # with tf.control_dependencies([print_op]):
            #     final = tf.identity(final)
            return final
        tensor = Lambda(elementwise_contextnet)(occurences)
        mask = Input(shape=(None,), dtype=tf.int8)
        output = PartialMean()([tensor, mask])
        # print(f"Output is {output}")
        super().__init__(inputs=[occurences, mask], outputs=output)
        self._trainable_weights = contextnet.trainable_weights

    # def __call__(self, *args, **kwargs):
    #     print(f"Call called with {args} and {kwargs}")
    #     return super().__call__(*args, **kwargs)
