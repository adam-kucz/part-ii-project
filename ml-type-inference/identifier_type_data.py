"""TODO: describe"""
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from dpu_utils.mlutils.vocabulary import Vocabulary
from dpu_utils.mlutils.chartensorizer import CharTensorizer

DATA_DIR = "~/synced/part-ii-project/data/sets/pairs_trivial"
TRAIN_PATH = DATA_DIR + "/train.csv"
VALIDATE_PATH = DATA_DIR + "/validate.csv"

PERCENTAGE: float = 1.0  # 0.75
IDENTIFIER_LENGTH: int = 12

CSV_COLUMN_NAMES = ['identifier', 'type']
# COLUMNS = ['c' + str(i) for i in range(IDENTIFIER_LENGTH)] + ['type']


# TODO: change to use tf.data
class DataLoader:
    """TODO: class docstring"""
    vocab: Vocabulary
    char_tensorizer: CharTensorizer
    train_ds: Tuple
    validate_ds: Tuple

    def __init__(self):
        self.vocab = Vocabulary()
        self.char_tensorizer = CharTensorizer(IDENTIFIER_LENGTH, False, False)

    def _parse_csv(self, filename):
        """TODO: method docstring"""
        dataframe = pd.read_csv(filename, names=CSV_COLUMN_NAMES, header=0)
        # TODO: remove magic strings
        # TODO: fix list, use numpy arrays if possible
        char_arr = np.array(list(
            self.char_tensorizer.tensorize_str(identifier)
            for identifier in dataframe.pop('identifier')))
        chars = dict(('char{}'.format(i), char_arr[:, i])
                     for i in range(IDENTIFIER_LENGTH))
        typs = dataframe.pop('type')
        return chars, typs

    def _as_dataset(self, chars, typs):
        """TODO: method docstring"""
        typs = typs.map(self.vocab.get_id_or_unk)
        return (chars, typs)

    def load_data(self):
        """TODO: method docstring"""
        train_chars, train_typs = self._parse_csv(TRAIN_PATH)
        included = 0
        for typ, count in Counter(train_typs).items():
            self.vocab.add_or_get_id(typ)
            included += count
            if included / len(train_typs) > PERCENTAGE:
                break
        self.train_ds = self._as_dataset(train_chars, train_typs)
        # print(self.train_ds)

        self.validate_ds = self._as_dataset(*self._parse_csv(VALIDATE_PATH))
        # print(self.validate_ds)

        print(self.num_classes, "classes with", len(train_typs), "examples")

    @property
    def num_chars_in_vocabulary(self) -> int:
        """TODO: docstring for num_chars_in_vocabulary"""
        return self.char_tensorizer.num_chars_in_vocabulary()

    @property
    def num_classes(self) -> int:
        """TODO: docstring for num_classes"""
        return len(self.vocab)

    def training_data(self, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        print("train_ds:\n", self.train_ds)
        print("batch_size:\n", batch_size)

        dataset = tf.data.Dataset.from_tensor_slices(self.train_ds)
        # print("shape:\n", dataset.shape)

        # Shuffle, repeat, and batch the examples.
        assert batch_size is not None, "error: batch_size is None"  # nosec
        dataset = dataset.shuffle(1000).batch(batch_size)
        print("Dataset:\n{}".format(dataset))
        return dataset

    def validation_data(self, batch_size):
        """An input function for validation"""
        # Convert the inputs to a Dataset.
        print("validate_ds:\n", self.validate_ds)
        print("batch_size:\n", batch_size)

        dataset = tf.data.Dataset.from_tensor_slices(self.validate_ds)

        # Batch the data.
        assert batch_size is not None, "error: batch_size is None"  # nosec
        return dataset.batch(batch_size)

    def prediction_data(self, features, batch_size):
        """An input function for prediction"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(dict(features))

        # Returned batched Dataset.
        assert batch_size is not None, "error: batch_size is None"  # nosec
        return dataset.batch(batch_size)
