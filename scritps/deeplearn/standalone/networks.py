from pathlib import Path
from typing import Any, List, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Input

from ..modules.charcnn import CharCNN
from ..modules.contextnet import ContextNet
from .standard import CategoricalStandalone
from ..data_ops.data_interface import CsvReader
from ..data_ops.data_transformers import CategoricalIndex

__all__ = ['FullCharCNN', 'FullContextNet']


class FullCharCNN(CategoricalStandalone):
    def __init__(self, vocab_path: Path, batch_size: int,
                 params: Mapping[str, Any], out_dir: Path,
                 run_name: str = 'default'):
        vocab: List[str] = vocab_path.read_text().splitlines()
        dataset_producer = CsvReader((tf.string,), (tf.string,),
                                     CategoricalIndex(vocab, unk=True),
                                     batch_size)
        identifier = Input(shape=(), dtype=tf.string)
        outputs = CharCNN(params)(identifier)
        super().__init__('charcnn', dataset_producer, len(vocab) + 1,
                         identifier, outputs, out_dir, run_name)


class FullContextNet(CategoricalStandalone):
    def __init__(self, vocab_path: Path, batch_size: int,
                 params: Mapping[str, Any], out_dir: Path,
                 run_name: str = 'default'):
        vocab: List[str] = vocab_path.read_text().splitlines()
        ctx_len = params['ctx_len']
        dataset_producer = CsvReader((tf.string,) * ctx_len,
                                     (tf.string,),
                                     CategoricalIndex(vocab, unk=True),
                                     batch_size)
        tokens = tuple(Input(shape=(), dtype=tf.string)
                       for i in range(ctx_len))
        outputs = ContextNet(params)(tokens)
        super().__init__('contextnet', dataset_producer, len(vocab) + 1,
                         tokens, outputs, out_dir, run_name)