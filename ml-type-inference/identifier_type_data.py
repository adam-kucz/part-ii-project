"""TODO: identifier_type_data module docstring"""
import os
from typing import Any, List, Tuple

import tensorflow as tf
from tensorflow.feature_column import (  # pylint: disable=import-error
    categorical_column_with_vocabulary_list,
    indicator_column)
from dpu_utils.mlutils.chartensorizer import CharTensorizer

DIR = "/home/acalc79/synced/part-ii-project/data/sets/pairs"
VOCABULARY_PATH = os.path.join(DIR, "vocab.csv")


class DataLoader:
    """TODO: class docstring"""
    char_tensorizer: CharTensorizer
    vocab: List[str]
    char_col_names: Tuple[str]
    typ_col: Any  # TODO: figure out if we can assign correct type
    char_cols: Tuple

    def __init__(self, identifier_length):
        self.char_tensorizer = CharTensorizer(identifier_length, False, False)
        with open(VOCABULARY_PATH, 'r') as vocabfile:
            self.vocab = vocabfile.readlines()
        self._prepare_columns()

    def _prepare_columns(self):
        self.char_col_names = tuple('char{}'.format(i)
                                    for i in range(self.max_str_len))

        self.typ_col = categorical_column_with_vocabulary_list(
            'type', self.vocab, default_value=len(self.vocab))

        def char_col(name):
            return indicator_column(categorical_column_with_vocabulary_list(
                name, self.char_tensorizer.__ALPHABET))

        self.char_cols = tuple(char_col(name) for name in self.char_col_names)

    def _str_to_chars(self, string: tf.Tensor) -> tf.Tensor:
        num_cols = self.char_tensorizer.max_char_length
        substr = tf.substr(string, 0, num_cols)
        char_sparse = tf.string_split(tf.expand_dims(substr, axis=0), '')
        return tf.sparse.to_dense(
            tf.sparse.reset_shape(char_sparse, (1, num_cols)), '')[0]

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
            handle, (tf.string, tf.string), ([batch_size], [batch_size]))
        chars, typ = iterator.get_next()
        features = {'type': typ}
        for i, name in enumerate(self.char_col_names):
            features[name] = chars[i]
        char_input = tf.feature_column.input_layer(features, self.char_cols)
        labels = tf.feature_column.input_layer(features, self.typ_col)
        return handle, char_input, labels

    @property
    def num_chars_in_vocabulary(self) -> int:
        """TODO: docstring for num_chars_in_vocabulary"""
        return self.char_tensorizer.num_chars_in_vocabulary()

    @property
    def max_str_len(self) -> int:
        """TODO: docstring for num_chars_in_vocabulary"""
        return self.char_tensorizer.max_char_length
