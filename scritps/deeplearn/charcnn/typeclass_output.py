from typing import Mapping

import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Sequential

from ..abstract.modules import Parametrized

__all__ = ['TypeclassOutput']


# compile with
# optimizer=tf.train.AdamOptimizer()
# loss=tf.keras.losses.CategoricalCrossentropy()
# metrics=[tf.keras.metrics.CategoricalAccuracy()]
class TypeclassOutput(Sequential, Parametrized):
    _class_num: int

    def __init__(self, class_num: int):
        self._class_num = class_num
        self.add(Dense(self._class_num, activation=Softmax()))

    @property
    def params(self) -> Mapping[str, int]:
        return {'classnum': self._class_num}
