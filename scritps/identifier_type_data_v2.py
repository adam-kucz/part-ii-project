"""TODO: describe"""
from collections import Counter
from typing import Dict, Tuple

import tensorflow as tf
from dpu_utils.mlutils.vocabulary import Vocabulary
from dpu_utils.mlutils.chartensorizer import CharTensorizer

DATA_DIR = "~/synced/part-ii-project/data/sets/pairs_trivial"
TRAIN_PATH = DATA_DIR + "/train.csv"
VALIDATE_PATH = DATA_DIR + "/validate.csv"

PERCENTAGE: float = 1.0  # 0.75
IDENTIFIER_LENGTH: int = 12

FIELD_DEFAULTS = [[], []]
COLUMNS = ['c' + str(i) for i in range(IDENTIFIER_LENGTH)] + ['type']


class DataLoader:
    """TODO: class docstring"""
    vocab: Vocabulary
    char_tensorizer: CharTensorizer
    train_ds: tf.data.Dataset
    validate_ds: tf.data.Dataset

    def __init__(self):
        self.vocab = Vocabulary()
        self.char_tensorizer = CharTensorizer(IDENTIFIER_LENGTH, False, False)

    def _parse_line(self, line):
        # Decode the line into its fields
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
        fields = self.char_tensorizer.tensorize_str(fields[0]) + fields[1]

        # Pack the result into a dictionary
        features = dict(zip(COLUMNS, fields))

        # Separate the label from the features
        label = features.pop('type')

        return features, label

    def load_data(self, y_name='type'):
        """TODO: method docstring"""
        self.train_ds = tf.data.TextLineDataset(TRAIN_PATH)\
                               .skip(1)\
                               .map(self._parse_line)
        self.validate_ds = tf.data.TextLineDataset(VALIDATE_PATH)\
                                  .skip(1)\
                                  .map(self._parse_line)
        return
        train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
        train_x, train_y = train, train.pop(y_name)
        self.train_x = train_x.applymap(self.char_tensorizer.tensorize_str)
        included = 0
        for typ, count in Counter(train_y).items():
            self.vocab.add_or_get_id(typ)
            included += count
            if included / len(train_y) > PERCENTAGE:
                break
        self.train_y = train_y.map(self.vocab.get_id_or_unk)

        validate = pd.read_csv(VALIDATE_PATH, names=CSV_COLUMN_NAMES, header=0)
        validate_x, validate_y = validate, validate.pop(y_name)
        self.validate_x =\
            validate_x.applymap(self.char_tensorizer.tensorize_str)
        self.validate_y = validate_y.map(self.vocab.get_id_or_unk)
        print(str(self.num_classes) + " of classes with " +
              str(len(train_y)) + " examples")

    @property
    def num_chars_in_vocabulary(self) -> int:
        return self.char_tensorizer.num_chars_in_vocabulary()

    @property
    def num_classes(self) -> int:
        """TODO: docstring"""
        return len(self.vocab)

    # def get_data(self) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict]]:
    #     """
    #     Returns the types dataset

    #     as (train_x, train_y), (validate_x, validate_y).
    #     """
#     return (self.train_x, self.train_y), (self.validate_x, self.validate_y)

    def train_input_fn(self, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        print("train_ds:\n" + str(self.train_ds))
        print("batch_size:\n" + str(batch_size))

        # Shuffle, repeat, and batch the examples.
        # TODO: change to epochs recognized through exceptions
        assert batch_size is not None, "error: batch_size is None"  # nosec
        return self.train_ds.shuffle(1000).repeat().batch(batch_size)

    def validate_input_fn(self, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        print("validate_ds:\n" + str(self.validate_ds))
        print("batch_size:\n" + str(batch_size))

        # Batch the data.
        assert batch_size is not None, "error: batch_size is None"  # nosec
        return self.validate_ds.batch(batch_size)

    def predict_input_fn(self, features, batch_size):
        """An input function for evaluation or prediction"""
        features = dict(features)
        inputs = features

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "error: batch_size is None"  # nosec
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset
