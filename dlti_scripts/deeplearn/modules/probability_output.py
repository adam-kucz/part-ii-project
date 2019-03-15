from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Sequential

__all__ = ['ProbabilityOutput']


class ProbabilityOutput(Sequential):
    def __init__(self, class_num: int):
        super().__init__()
        self.add(Dense(class_num))
        self.add(Softmax())
