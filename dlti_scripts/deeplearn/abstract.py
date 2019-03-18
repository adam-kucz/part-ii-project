from abc import ABC, abstractmethod
from enum import auto, Flag
from pathlib import Path
from typing import Optional

import tensorflow as tf

__all__ = ['DataMode', 'DataReader']


class DataMode(Flag):
    """
    Mode for reading a dataset

    INPUTS - inputs are present
    LABELS - labels are present
    BATCH - dataset batched
    SHUFFLE - dataset will be shuffled
    TRAIN - batched, inputs and labels with shuffling
    TEST - batched, inputs and labels, no shuffling
    PREDICT - batched, inputs only, no shuffling, one pass only
    """
    INPUTS = auto()
    LABELS = auto()
    BATCH = auto()
    SHUFFLE = auto()
    ONEPASS = auto()
    BASIC = INPUTS | LABELS | BATCH
    TRAIN = BASIC | SHUFFLE
    VALIDATE = BASIC
    TEST = BASIC | ONEPASS
    # pylint doesn't seem to know that flags can be negated
    PREDICT = TEST & ~LABELS  # pylint: disable=invalid-unary-operand-type


class DataReader(ABC):
    @abstractmethod
    def __call__(self, path: Path,
                 mode: Optional[DataMode]) -> tf.data.Dataset:
        pass
