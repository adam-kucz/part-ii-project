"""TODO: describe"""
from collections import Counter
from typing import Dict, Tuple

import pandas as pd
import tensorflow as tf
from dpu_utils.mlutils.vocabulary import Vocabulary


CSV_COLUMN_NAMES = ['identifier', 'type']
TRAIN_PATH = "~/synced/part-ii-project/data/sets/train.csv"
VALIDATE_PATH = "~/synced/part-ii-project/data/sets/validate.csv"

PERCENTAGE: float = 0.99


class DataLoader:
    """TODO: class docstring"""
    vocab: Vocabulary = Vocabulary()
    train_x: pd.DataFrame
    train_y: pd.Series
    validate_x: pd.DataFrame
    validate_y: pd.Series

    def load_data(self, y_name='type'):
        """TODO: method docstring"""
        train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
        self.train_x, train_y = train, train.pop(y_name)
        included = 0
        for typ, count in Counter(train_y).items():
            self.vocab.add_or_get_id(typ)
            included += count
            if included / len(train_y) > PERCENTAGE:
                break
        self.train_y = train_y.map(self.vocab.get_id_or_unk)

        validate = pd.read_csv(VALIDATE_PATH, names=CSV_COLUMN_NAMES, header=0)
        self.validate_x, validate_y = validate, validate.pop(y_name)
        self.validate_y = validate_y.map(self.vocab.get_id_or_unk)

    def get_num_classes(self) -> int:
        """TODO: docstring"""
        return len(self.vocab)

    def get_data(self) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict]]:
        """
        Returns the types dataset

        as (train_x, train_y), (validate_x, validate_y).
        """
        return (self.train_x, self.train_y), (self.validate_x, self.validate_y)

    def train_input_fn(self, features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the dataset.
        return dataset

    def eval_input_fn(self, features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

            # Convert the inputs to a Dataset.
            dataset = tf.data.Dataset.from_tensor_slices(inputs)

            # Batch the examples
            assert batch_size is not None, "error: batch_size is None"  # nosec
            dataset = dataset.batch(batch_size)

            # Return the dataset.
            return dataset
