from pathlib import Path
from typing import Any, Callable, List, Mapping

import tensorflow as tf

from .charcnn import CharCNN
from ..data_ops.data_interface import CsvReader, PairInterface
from ..data_ops.data_transformers import PairProc, StrEnc, Categorical
from .network_assembly import Modular
from .trainer import Trainer
from .typeclass_output import TypeclassOutput

__all__ = ['FullCharCNN']


class FullCharCNN(Trainer):
    def __init__(self,
                 vocab_path: Path,
                 identifier_length: int,
                 batch_size: int,
                 params: Mapping[str, Any],
                 out_dir: Path,
                 log: Callable[[str], None] = lambda _: None):
        vocab: List[str] = vocab_path.read_text().splitlines()
        dataset_producer = CsvReader((tf.string,), (tf.string,), batch_size)
        data_processor = PairProc(StrEnc(identifier_length),
                                  Categorical(vocab, unk=True))
        core_net = CharCNN(params, log)
        output_net = TypeclassOutput(len(vocab) + 1, log)
        super().__init__(dataset_producer,
                         PairInterface(),
                         Modular(data_processor, core_net, output_net, log),
                         out_dir)