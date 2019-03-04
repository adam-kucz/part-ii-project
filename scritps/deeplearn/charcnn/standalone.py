from pathlib import Path
from typing import Any, List, Mapping

import tensorflow as tf
from tensorflow.keras.models import Sequential

from .charcnn import CharCNN
from ..data_ops.data_interface import CsvReader
from ..data_ops.data_transformers import CategoricalOneHot
from .trainer import Trainer
from .typeclass_output import TypeclassOutput

__all__ = ['FullCharCNN']


class FullCharCNN(Trainer):
    def __init__(self,
                 vocab_path: Path,
                 params: Mapping[str, Any],
                 out_dir: Path):
        vocab: List[str] = vocab_path.read_text().splitlines()
        dataset_producer = CsvReader((tf.string,), (tf.string,),
                                     CategoricalOneHot(vocab, unk=True))
        super().__init__('charcnn',
                         dataset_producer,
                         Sequential([CharCNN(params),
                                     TypeclassOutput(len(vocab) + 1)]),
                         out_dir)
