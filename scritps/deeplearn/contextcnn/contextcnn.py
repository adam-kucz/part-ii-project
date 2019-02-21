from typing import Any, Callable, Mapping, Tuple

import tensorflow as tf

from ..util import merge_parametrized
from ..abstract.modules import CoreNet
from ..charcnn.charcnn import CharCNN

__all__ = ['contextcnn', 'ContextCNN']


def contextcnn(one_hot_chars: tf.Tensor,
               params: Mapping[str, Any],
               log: Callable[[str], None] = lambda _: None) -> tf.Tensor:
    return CharCNN(params, log)(one_hot_chars)


# TODO: consider inheriting from tf.layers.Layer or tf.keras.layers.Layer
class ContextCNN(CoreNet):
    def __init__(self,
                 params: Mapping[str, Any],
                 log: Callable[[str], None] = lambda _: None,
                 separate_scopes: bool = True):
        self.center_net = CharCNN(params['center'], log)
        self.context_net = CharCNN(params['context'], log)
        self.dense_params = params['aggregate']
        self.log = log
        self.separate_scopes = separate_scopes

    def __call__(self, tokens: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        self.log("Shapes of inputs: {}"
                 .format(tuple(inp.shape for inp in tokens)))
        central, rest = tokens[0], tokens[1:]

        with tf.name_scope("central"):
            central_tensor = self.center_net((central,))

        with tf.name_scope("context"):
            if self.separate_scopes:
                context_tensors = []
                for i, tensor in enumerate(rest):
                    with tf.name_scope("context{}".format(i)):
                        context_tensors.append(self.context_net(tensor))
            else:
                context_tensors = list(self.context_net(token)
                                       for token in rest)

        with tf.name_scope("dense"):
            # TODO: replace deprecated
            flat = tf.layers.flatten(
                tf.concat([central_tensor] + context_tensors, 1))
            tensor = self._dense(flat)

        return tensor

    def _dense(self, tensor: tf.Tensor) -> tf.Tensor:
        for i, layer_params in enumerate(self.dense_params):
            layer = tf.layers.Dense(units=layer_params['units'],
                                    activation=tf.nn.relu)
            if self.separate_scopes:
                with tf.name_scope('dense{}'.format(i)):
                    tensor = layer(tensor)
            else:
                tensor = layer(tensor)
            self.log("Shape of tensor after dense {}: {}"
                     .format(i, tensor.shape))
        return tensor

    @property
    def params(self):
        params = {'aggregate': self.dense_params}
        params.update(merge_parametrized(('center', self.center_net),
                                         ('context', self.context_net)))
        return params
