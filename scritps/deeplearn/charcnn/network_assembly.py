from functools import reduce
import operator as op
from typing import Any, Callable, Mapping, Tuple

import tensorflow as tf

from ..util import merge_parametrized
from ..abstract.modules import CoreNet, DataProcessor, FullNet, OutputNet

__all__ = ['modular', 'Modular']


def modular(data_tensor, data_trans, core_net, output_layer):
    return Modular(data_trans, core_net, output_layer)(data_tensor)


class Modular(FullNet):
    _out: OutputNet

    def __init__(self,
                 data_trans: DataProcessor,
                 core_net: CoreNet,
                 output_layer: OutputNet,
                 log: Callable[[str], None] = lambda _: None):
        self.data = data_trans
        self.core = core_net
        self._out = output_layer
        self.log = log

    def __call__(self, data_tensors: Tuple[tf.Tensor, ...]) -> OutputNet:
        inputs, labels = self.data(data_tensors)
        outputs = self.core(inputs)
        result = self._out(outputs, labels)

        total_params = reduce(
            op.add,
            (reduce(op.mul, (dim.value for dim in var.get_shape()), 1)
             for var in tf.trainable_variables()),
            0)
        self.log("Total number of trainable parameters: {}"
                 .format(total_params))
        return result

    @property
    def out(self) -> OutputNet:
        return self._out

    @property
    def params(self) -> Mapping[str, Any]:
        return merge_parametrized(('data', self.data),
                                  ('core', self.core),
                                  ('out', self.out))
