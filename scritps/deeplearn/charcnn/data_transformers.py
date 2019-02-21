import string
from typing import Any, Callable, Iterable, Mapping, Tuple

import tensorflow as tf
from tensorflow.feature_column import (  # pylint: disable=import-error
    categorical_column_with_vocabulary_list,
    indicator_column,
    input_layer)

from ..util import merge_parametrized
from ..abstract.modules import DataProcessor, Parametrized

__all__ = ['categorical', 'Categorical',
           'str_enc', 'StrEnc',
           'pair_trans', 'PairTrans']


def categorical(tensor: tf.Tensor, alphabet: Iterable[str], unk=False):
    return Categorical(alphabet, unk)(tensor)


class Categorical(Parametrized):
    def __init__(self, alphabet, unk=False):
        self.alphabet = alphabet
        self.class_num = len(alphabet)
        self.oov = 1 if unk else 0

    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        colname = 'col'
        col = indicator_column(
            categorical_column_with_vocabulary_list(colname,
                                                    self.alphabet,
                                                    num_oov_buckets=self.oov))
        return input_layer({colname: tensor}, col)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'alphabet': self.alphabet,
                'class_num': self.class_num + self.oov}



def str_enc(strs: tf.Tensor, str_length: int):
    return StrEnc(str_length)(strs)


class StrEnc(Parametrized):
    def __init__(self, str_length: int):
        self.str_length = str_length

    def __call__(self, strs: tf.Tensor) -> tf.Tensor:
        tensor = self._char_tokenise(strs)
        # print("StrEnc, tensor.shape: {}".format(tensor.shape))
        alphabet = string.ascii_lowercase + string.ascii_uppercase +\
                   string.digits + string.punctuation  # noqa: E127
        cat_layer = Categorical(alphabet)
        # print("StrEnc, tensor.shape: {}".format(tensor.shape))
        return tf.stack([cat_layer(tensor[:, i])
                         for i in range(self.str_length)],
                        axis=1)

    def _char_tokenise(self, strs: tf.Tensor):
        # print("char tokenising, strs: {}".format(strs))
        truncated = tf.substr(strs, 0, self.str_length)
        # print("char tokenising, truncated: {}".format(truncated))
        sparse_chars = tf.string_split(truncated, '')
        # print("char tokenising, sparse_chars: {}".format(sparse_chars))
        sparse_padded = tf.sparse.reset_shape(
            sparse_chars, (sparse_chars.dense_shape[0], self.str_length))
        # print("char tokenising, sparse_padded: {} (dense shape: {})"
        # .format(sparse_padded, sparse_padded.dense_shape))
        return tf.sparse.to_dense(sparse_padded, '')

    @property
    def params(self) -> Mapping[str, Any]:
        return {'strlen': self.str_length}


def pair_trans(data_tensor, input_trans, label_trans):
    return PairTrans(input_trans, label_trans)(data_tensor)


# TODO: generalize transformer types
class PairTrans(DataProcessor):
    def __init__(self,
                 input_trans: Callable[[tf.Tensor], tf.Tensor],
                 label_trans: Callable[[tf.Tensor], tf.Tensor]):
        self.input_trans = input_trans
        self.label_trans = label_trans

    def __call__(self, data_tensor: Tuple[tf.Tensor, ...])\
            -> Tuple[tf.Tensor, tf.Tensor]:
        return (self.input_trans(data_tensor[0]),
                self.label_trans(data_tensor[1]))

    @property
    def params(self):
        return merge_parametrized(('inputs', self.input_trans),
                                  ('labels', self.label_trans))
