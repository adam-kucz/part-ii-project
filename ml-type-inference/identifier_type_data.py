"""TODO: identifier_type_data module docstring"""
import os
import string
from typing import Any, List, Tuple

import tensorflow as tf
from tensorflow.feature_column import (  # pylint: disable=import-error
    categorical_column_with_vocabulary_list,
    indicator_column)

DIR = "/home/acalc79/synced/part-ii-project/data/sets/pairs"
VOCABULARY_PATH = os.path.join(DIR, "vocab.txt")


class DataLoader:
    """TODO: class docstring"""
    vocab: List[str]
    char_col_names: Tuple[str]
    typ_col: Any  # TODO: figure out if we can assign correct type
    char_cols: Tuple
    alphabet_len: int

    def __init__(self, identifier_length):
        self.identifier_length = identifier_length
        with open(VOCABULARY_PATH, 'r') as vocabfile:
            self.vocab = vocabfile.read().splitlines()
        print("Types selected: {}".format(self.vocab[:50]))
        self._prepare_columns()

    def _prepare_columns(self):
        self.char_col_names = tuple('char{}'.format(i)
                                    for i in range(self.identifier_length))
        self.typ_col = indicator_column(
            categorical_column_with_vocabulary_list(
                'type', self.vocab, default_value=len(self.vocab)))
        alphabet = string.ascii_lowercase + string.ascii_uppercase +\
                   string.digits + string.punctuation  # noqa: E127
        self.alphabet_len = len(alphabet)

        def char_col(name):
            return indicator_column(categorical_column_with_vocabulary_list(
                name, alphabet))

        self.char_cols = tuple(char_col(name) for name in self.char_col_names)

    def _str_to_chars(self, str_input: tf.Tensor) -> tf.Tensor:
        num_cols = self.identifier_length
        substr = tf.substr(str_input, 0, num_cols)
        char_sparse = tf.string_split(tf.expand_dims(substr, axis=0), '')
        result = tf.sparse.to_dense(
            tf.sparse.reset_shape(char_sparse, (1, num_cols)), '')[0]
        # print("Input shape: {}, result.shape: {}"
        #       .format(str_input.shape, result.shape))
        return result

    def read_dataset(self, filename, labelled=True):
        """TODO: method docstring"""
        dataset = tf.data.experimental.CsvDataset(
            filename,
            (tf.string, tf.string) if labelled else (tf.string),
            header=True)

        def transform(idn, typ):
            return (self._str_to_chars(idn), typ)

        return dataset.map(transform if labelled else self._str_to_chars)

    def handle_to_input_tensors(self, batch_size):
        """TODO: process_dataset docstring"""
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle,
            (tf.string, tf.string),
            ([None, self.identifier_length], [None]))
        chars, typ = iterator.get_next()
        # print("Chars shape: {}".format(chars.shape))
        features = {'type': typ}
        for i, name in enumerate(self.char_col_names):
            features[name] = chars[:, i]
        char_input = tf.reshape(
            tf.feature_column.input_layer(features, self.char_cols),
            (-1, self.identifier_length, self.alphabet_len))
        labels = tf.feature_column.input_layer(features, self.typ_col)
        unks = tf.to_float(tf.equal(tf.reduce_sum(labels, 1), 0))
        labels_with_unks = tf.concat((labels, tf.expand_dims(unks, 1)), 1)
        return handle, char_input, labels_with_unks
