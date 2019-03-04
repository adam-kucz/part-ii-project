from typing import Mapping

from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Sequential

__all__ = ['TypeclassOutput']


# compile with
# optimizer=tf.train.AdamOptimizer()
# loss=tf.keras.losses.CategoricalCrossentropy()
# metrics=[tf.keras.metrics.CategoricalAccuracy()]
class TypeclassOutput(Sequential):
    _class_num: int

    def __init__(self, class_num: int):
        super().__init__()
        self._class_num = class_num
        self.add(Dense(self._class_num))
        self.add(Softmax())

    @property
    def params(self) -> Mapping[str, int]:
        return {'classnum': self._class_num}
