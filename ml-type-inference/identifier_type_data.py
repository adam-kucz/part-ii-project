"""TODO: describe"""
from typing import Any, List, Tuple

import tensorflow as tf
from tensorflow.feature_column import (  # pylint: disable=import-error
    categorical_column_with_vocabulary_list,
    indicator_column)
from dpu_utils.mlutils.chartensorizer import CharTensorizer

DATA_DIR = "/home/acalc79/synced/part-ii-project/data/sets/pairs"
VOCABULARY_PATH = DATA_DIR + "/vocab.csv"
# TRAIN_PATH = DATA_DIR + "/train.csv"
# VALIDATE_PATH = DATA_DIR + "/validate.csv"


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

    def process_dataset(self, dataset, batch_size, labelled=True):
        iterator = dataset.batch(batch_size).make_initializable_iterator()
        chars = iterator.get_next()
        if labelled:
            chars, typ = chars
        features = {}
        for i, name in enumerate(self.char_col_names):
            features[name] = chars[i]
        char_input = tf.feature_column.input_layer(features, self.char_cols)
        if not labelled:
            return iterator, char_input
        features['type'] = typ
        labels = tf.feature_column.input_layer(features, self.typ_col)
        return iterator, char_input, labels

    @property
    def num_chars_in_vocabulary(self) -> int:
        """TODO: docstring for num_chars_in_vocabulary"""
        return self.char_tensorizer.num_chars_in_vocabulary()

    @property
    def max_str_len(self) -> int:
        """TODO: docstring for num_chars_in_vocabulary"""
        return self.char_tensorizer.max_char_length
