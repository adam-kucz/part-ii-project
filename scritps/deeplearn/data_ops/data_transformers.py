import string
from typing import Iterable

import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.contrib.lookup import index_table_from_tensor, LookupInterface

__all__ = ['CategoricalIndex', 'CategoricalOneHot', 'CharLevel', 'StrEnc']


class CategoricalIndex(Lambda):
    def __init__(self, alphabet: Iterable[str], unk=False):
        self.table: LookupInterface\
            = index_table_from_tensor(tf.constant(tuple(alphabet)),
                                      num_oov_buckets=1 if unk else 0)
        super().__init__(self.table.lookup)


class OneHot(Lambda):
    def __init__(self, depth):
        super().__init__(lambda t: tf.one_hot(t, depth))


class CategoricalOneHot(Sequential):
    def __init__(self, alphabet: Iterable[str], unk: bool = False):
        super().__init__()
        alphabet = tuple(alphabet)
        self.add(CategoricalIndex(alphabet, unk))
        self.add(OneHot(len(alphabet) + (1 if unk else 0)))


class CharLevel(Lambda):
    def __init__(self, num_chars: int):
        self.num_chars = num_chars
        super().__init__(self._char_tokenise)

    def _char_tokenise(self, strs: tf.Tensor) -> tf.Tensor:
        truncated = tf.substr(strs, 0, self.num_chars)
        sparse_chars = tf.string_split(truncated, '')
        sparse_padded = tf.sparse.reset_shape(
            sparse_chars, (sparse_chars.dense_shape[0], self.num_chars))
        return tf.sparse.to_dense(sparse_padded, '')


class StrEnc(Sequential):
    def __init__(self, str_length: int):
        super().__init__()
        self.str_length = str_length
        self.alphabet = string.ascii_lowercase + string.ascii_uppercase +\
                        string.digits + string.punctuation  # noqa: E127
        self.add(CharLevel(str_length))
        self.add(CategoricalOneHot(self.alphabet))
