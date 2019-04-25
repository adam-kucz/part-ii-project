import string
from typing import Iterable

import tensorflow as tf
from tensorflow.keras.layers import Lambda, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.contrib.lookup import index_table_from_tensor, LookupInterface


class CategoricalIndex(Lambda):
    def __init__(self, alphabet: Iterable[str], unk=False):
        table_tensor = tf.constant(tuple(alphabet))
        self.table: LookupInterface\
            = index_table_from_tensor(table_tensor,
                                      num_oov_buckets=1 if unk else 0)
        super().__init__(self.table.lookup,
                         output_shape=self.compute_output_shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class OneHot(Lambda):
    def __init__(self, depth):
        self.depth = depth
        super().__init__(lambda t: tf.one_hot(t, depth),
                         output_shape=self.compute_output_shape)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.depth,)


class CategoricalOneHot(Sequential):
    def __init__(self, alphabet: Iterable[str], unk: bool = False):
        alphabet = tuple(alphabet)
        super().__init__()
        self.add(InputLayer(input_shape=(), dtype=tf.string))
        self.add(CategoricalIndex(alphabet, unk))
        self.add(OneHot(len(alphabet) + (1 if unk else 0)))


class CharLevel(Lambda):
    def __init__(self, num_chars: int):
        self.num_chars = num_chars
        super().__init__(self._char_tokenise,
                         output_shape=self.compute_output_shape)

    def _char_tokenise(self, strs: tf.Tensor) -> tf.Tensor:
        truncated = tf.strings.substr(strs, 0, self.num_chars)
        sparse_chars = tf.string_split(truncated, '')
        sparse_padded = tf.sparse.reset_shape(
            sparse_chars, (sparse_chars.dense_shape[0], self.num_chars))
        tensor = tf.sparse.to_dense(sparse_padded, '')
        tensor = tf.ensure_shape(tensor, (None, self.num_chars))
        return tensor

    def compute_output_shape(self, input_shape):
        return input_shape + (self.num_chars,)


class StringEncoder(Sequential):
    def __init__(self, str_length: int):
        self.str_length = str_length
        # '\n' might be useful because it is preserved as a token by parso
        self.alphabet = string.ascii_lowercase + string.ascii_uppercase +\
                        string.digits + string.punctuation + '\n'  # noqa: E127
        super().__init__()
        self.add(InputLayer(input_shape=(), dtype=tf.string))
        self.add(CharLevel(str_length))
        self.add(CategoricalOneHot(self.alphabet))
