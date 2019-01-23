"""TODO: describe"""
from typing import Dict, Tuple

import pandas as pd
import tensorflow as tf
import dpu_utils.mlutils.vocabulary as vocab


CSV_COLUMN_NAMES = ['identifier', 'type']
TRAIN_PATH = "~/synced/part-ii-project/data/sets/train.csv"
VALIDATE_PATH = "~/synced/part-ii-project/data/sets/validate.csv"


class DataLoader:
    """TODO: class docstring"""
    vocab: vocab.Vocabulary

    def load_data(self, y_name='type'):
        """TODO: method docstring"""
        train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
        self.train_x, self.train_y = train, train.pop(y_name)
        print("train y is of type " + str(type(self.train_y)) +
              " and len " + str(len(self.train_y)))

        test = pd.read_csv(VALIDATE_PATH, names=CSV_COLUMN_NAMES, header=0)
        self.validate_x, self.validate_y = test, test.pop(y_name)

    def get_num_classes(self) -> int:
        return 0

    def get_data(self) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict]]:
        """
        Returns the types dataset

        as (train_x, train_y), (validate_x, validate_y).
        """
        return ({}, {}), ({}, {})

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
