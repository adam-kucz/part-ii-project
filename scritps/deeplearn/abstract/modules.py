from abc import ABC, abstractmethod
from enum import auto, Flag, unique
from pathlib import Path
from typing import Optional

import tensorflow as tf

__all__ = ['DataMode', 'DataReader']


# TODO: rename and move to data folder
@unique
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
    TRAIN = INPUTS | LABELS | BATCH | SHUFFLE
    TEST = INPUTS | LABELS | BATCH
    PREDICT = INPUTS | BATCH | ONEPASS


class DataReader(ABC):
    @abstractmethod
    def __call__(self, path: Path,
                 mode: Optional[DataMode]) -> tf.data.Dataset:
        pass
