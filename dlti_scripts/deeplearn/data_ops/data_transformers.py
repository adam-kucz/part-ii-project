import string
from typing import Iterable

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.contrib.lookup import index_table_from_tensor, LookupInterface

__all__ = ['CategoricalIndex', 'CategoricalOneHot',
           'CharLevel', 'StringEncoder']


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


class CategoricalOneHot(Model):
    def __init__(self, alphabet: Iterable[str], unk: bool = False):
        alphabet = tuple(alphabet)
        strs = Input(shape=(), dtype=tf.string)
        indices = CategoricalIndex(alphabet, unk)(strs)
        one_hot = OneHot(len(alphabet) + (1 if unk else 0))(indices)
        super().__init__(strs, one_hot)


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


class StringEncoder(Model):
    def __init__(self, str_length: int):
        self.str_length = str_length
        # '\n' might be useful because it is preserved as a token by parso
        self.alphabet = string.ascii_lowercase + string.ascii_uppercase +\
                        string.digits + string.punctuation + '\n'  # noqa: E127
        strs = Input(shape=(), dtype=tf.string)
        chars = CharLevel(str_length)(strs)
        char_one_hot = CategoricalOneHot(self.alphabet)(chars)
        super().__init__(strs, char_one_hot)
