from pathlib import Path
from typing import Any, Callable, List, Mapping

import tensorflow as tf
from tensorflow.keras.models import Sequential

from .charcnn import CharCNN
from ..data_ops.data_interface import CsvReader, PairInterface
from ..data_ops.data_transformers import PairProc, StrEnc, Categorical
from .trainer import Trainer
from .typeclass_output import TypeclassOutput

__all__ = ['FullCharCNN']


class FullCharCNN(Trainer):
    def __init__(self,
                 vocab_path: Path,
                 identifier_length: int,
                 batch_size: int,
                 params: Mapping[str, Any],
                 out_dir: Path):
        # merge start
        vocab: List[str] = vocab_path.read_text().splitlines()
        dataset_producer = CsvReader((tf.string,), (tf.string,), batch_size)
        data_processor = PairProc(StrEnc(identifier_length),
                                  Categorical(vocab, unk=True))
        # merge end
        core_net = CharCNN(params)
        output_net = TypeclassOutput(len(vocab) + 1)
        super().__init__(dataset_producer,
                         Sequential(core_net, output_net),
                         out_dir)
