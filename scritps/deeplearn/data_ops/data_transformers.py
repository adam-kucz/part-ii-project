from pathlib import Path
import string
from typing import (Any, Callable, Iterable, List,
                    Mapping, Optional, Tuple, TypeVar)

import tensorflow as tf
from tensorflow.feature_column import (  # pylint: disable=import-error
    categorical_column_with_vocabulary_list,
    indicator_column,
    input_layer)

from ..util import merge_parametrized
from ..abstract.modules import (DataMode, DataReader, Processor,
                                Parametrized, Tensors)

__all__ = ['Categorical', 'CsvReader', 'StrEnc',
           'PairProc', 'HeadTailProc', 'InitLastProc']


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')


class CsvReader(DataReader):
    def __init__(self, input_shape, label_shape, batch_size: int) -> None:
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.label_shape = label_shape

    def __call__(self, path: Path, in_mode: Optional[DataMode] = None)\
            -> tf.data.Iterator:
        mode: DataMode = in_mode or DataMode.TRAIN
        shape: Tuple[tf.DType, ...] = ()
        if mode & DataMode.INPUTS:
            shape += self.input_shape
        if mode & DataMode.LABELS:
            shape += self.label_shape
        return tf.data.experimental.CsvDataset(str(path), shape, header=True)

        if mode & DataMode.TRAIN:
            dataset = dataset.shuffle(1000)
        if mode & DataMode.BATCH:
            dataset = dataset.batch(self.batch_size)

        if mode & DataMode.ONEPASS:
            return dataset.make_one_shot_iterator()
        return dataset.make_initializable_iterator()


class Categorical(Processor[A, List[int]]):
    def __init__(self, alphabet: Iterable[A], unk=False):
        self.alphabet = tuple(alphabet)
        self.oov = 1 if unk else 0

    def __call__(self, element: A) -> List[int]:
        indicator = [0] * self.class_num
        try:
            index = self.alphabet.index(element)
        except ValueError:
            if self.oov == 0:
                return indicator
            index = len(self.alphabet)
        indicator[index] = 1
        return indicator

    @property
    def class_num(self):
        return len(self.alphabet) + self.oov

    @property
    def params(self) -> Mapping[str, Any]:
        return {'alphabet': self.alphabet,
                'class_num': self.class_num + self.oov}


class StrEnc(Processor[str, List[List[int]]]):
    def __init__(self, str_length: int):
        self.str_length = str_length
        alphabet = string.ascii_lowercase + string.ascii_uppercase +\
                   string.digits + string.punctuation  # noqa: E127
        self.categorical = Categorical(alphabet)

    def __call__(self, strs_input: str) -> List[List[int]]:
        padded = strs_input[:self.str_length] +\
                 ' ' * (self.str_length - len(strs_input))  # noqa: E126
        return list(self.categorical(c) for c in padded)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'strlen': self.str_length}


class PairProc(Processor[Tuple[A, B], Tuple[C, D]]):
    def __init__(self,
                 fst_trans: Processor[A, C],
                 snd_trans: Processor[B, D]):
        self.fst_trans = fst_trans
        self.snd_trans = snd_trans

    def __call__(self, data: Tuple[A, B]) -> Tuple[C, D]:
        return (self.fst_trans(data[0]),
                self.snd_trans(data[1]))

    @property
    def params(self):
        return merge_parametrized(('fst', self.fst_trans),
                                  ('snd', self.snd_trans))


class HeadTailProc(Processor[List, List]):
    def __init__(self,
                 head_trans: Processor,
                 tail_trans: Processor[List, List]):
        self.head_trans = head_trans
        self.tail_trans = tail_trans

    def __call__(self, data: List) -> List:
        return [self.head_trans(data[0])] + self.tail_trans(data[1:])

    @property
    def params(self):
        return merge_parametrized(('head', self.head_trans),
                                  ('tail', self.tail_trans))


class InitLastProc(Processor[List, List]):
    def __init__(self,
                 init_trans: Processor[List, List],
                 last_trans: Processor):
        self.init_trans = init_trans
        self.last_trans = last_trans

    def __call__(self, data: List) -> List:
        return self.init_trans(data[:-1]) + [self.last_trans(data[-1])]

    @property
    def params(self):
        return merge_parametrized(('init', self.init_trans),
                                  ('last', self.last_trans))
