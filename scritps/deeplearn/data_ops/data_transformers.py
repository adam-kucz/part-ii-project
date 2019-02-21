import string
from typing import Any, Callable, Iterable, Mapping, Tuple

import tensorflow as tf
from tensorflow.feature_column import (  # pylint: disable=import-error
    categorical_column_with_vocabulary_list,
    indicator_column,
    input_layer)

from ..util import merge_parametrized
from ..abstract.modules import DataProcessor, Parametrized

__all__ = ['categorical', 'Categorical', 'str_enc', 'StrEnc', 'ProcTrans',
           'pair_proc', 'PairProc', 'HeadTailProc', 'InitLastProc']


Tensors = Tuple[tf.Tensor, ...]
Transformer = Callable[[Tensors], Tensors]
ProcessorOut = Tuple[Tensors, Tensors]


def categorical(tensor: Tensors,
                alphabet: Iterable[str], unk=False) -> Tensors:
    return Categorical(alphabet, unk)(tensor)


class Categorical(Parametrized):
    def __init__(self, alphabet, unk=False):
        self.alphabet = alphabet
        self.class_num = len(alphabet)
        self.oov = 1 if unk else 0

    def __call__(self, tensors: Tensors) -> Tensors:
        results = []
        for tensor in tensors:
            colname = 'col'
            col = indicator_column(
                categorical_column_with_vocabulary_list(
                    colname, self.alphabet, num_oov_buckets=self.oov))
            results.append(input_layer({colname: tensor}, col))
        return tuple(results)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'alphabet': self.alphabet,
                'class_num': self.class_num + self.oov}


def str_enc(strs: Tensors, str_length: int):
    return StrEnc(str_length)(strs)


class StrEnc(Parametrized):
    def __init__(self, str_length: int):
        self.str_length = str_length

    def __call__(self, strs_tensors: Tensors) -> Tensors:
        results = []
        for strs in strs_tensors:
            tensor = self._char_tokenise(strs)
            alphabet = string.ascii_lowercase + string.ascii_uppercase +\
                       string.digits + string.punctuation  # noqa: E127
            cat_layer = Categorical(alphabet)
            char_tensors = tuple(tensor[:, i] for i in range(self.str_length))
            results.append(tf.stack(cat_layer(char_tensors), axis=1))
        return tuple(results)

    def _char_tokenise(self, strs: tf.Tensor):
        truncated = tf.substr(strs, 0, self.str_length)
        sparse_chars = tf.string_split(truncated, '')
        sparse_padded = tf.sparse.reset_shape(
            sparse_chars, (sparse_chars.dense_shape[0], self.str_length))
        return tf.sparse.to_dense(sparse_padded, '')

    @property
    def params(self) -> Mapping[str, Any]:
        return {'strlen': self.str_length}


class ProcTrans(Parametrized):
    def __init__(self, processor: DataProcessor):
        self.proc = processor

    def __call__(self, tensors: Tensors) -> Tensors:
        init, tail = self.proc(tensors)
        return init + tail

    @property
    def params(self):
        return self.proc.params


def pair_proc(data_tensor, input_trans, label_trans):
    return PairProc(input_trans, label_trans)(data_tensor)


class PairProc(DataProcessor):
    def __init__(self, fst_trans: Transformer, snd_trans: Transformer):
        self.fst_trans = fst_trans
        self.snd_trans = snd_trans

    def __call__(self, data_tensor: Tensors) -> ProcessorOut:
        return (self.fst_trans(data_tensor[:1]),
                self.snd_trans(data_tensor[1:2]))

    @property
    def params(self):
        return merge_parametrized(('fst', self.fst_trans),
                                  ('snd', self.snd_trans))


class HeadTailProc(DataProcessor):
    def __init__(self, head_trans: Transformer, tail_trans: Transformer):
        self.head_trans = head_trans
        self.tail_trans = tail_trans

    def __call__(self, data_tensor: Tensors) -> ProcessorOut:
        return (self.head_trans(data_tensor[:1]),
                (self.tail_trans(data_tensor[1:]),))

    @property
    def params(self):
        return merge_parametrized(('head', self.head_trans),
                                  ('tail', self.tail_trans))


class InitLastProc(DataProcessor):
    def __init__(self, init_trans: Transformer, last_trans: Transformer):
        self.init_trans = init_trans
        self.last_trans = last_trans

    def __call__(self, data_tensor: Tensors) -> ProcessorOut:
        return (self.init_trans(data_tensor[:-1]),
                self.last_trans(data_tensor[-1:]))

    @property
    def params(self):
        return merge_parametrized(('init', self.init_trans),
                                  ('last', self.last_trans))
