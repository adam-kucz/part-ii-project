from pathlib import Path
from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Input

from ..modules.charcnn import CharCNN
from ..modules.contextnet import ContextNet
from ..modules.occurence_net import OccurenceNet
from .standard import VocabCategoricalStandalone
from ..data_ops.data_interface import RegularCsvReader, OccurenceCsvReader

__all__ = ['FullCharCNN', 'FullContextNet']


class FullCharCNN(VocabCategoricalStandalone):
    def __init__(self, vocab_path: Path, batch_size: int,
                 params: Mapping[str, Any], out_dir: Path,
                 run_name: str = 'default',
                 optimizer=tf.keras.optimizers.Adam()):
        dataset_producer = RegularCsvReader((tf.string,), (tf.string,),
                                            batch_size)
        identifier = Input(shape=(), dtype=tf.string)
        core = CharCNN(params)
        super().__init__('charcnn', dataset_producer, vocab_path,
                         identifier, core(identifier), core,
                         out_dir, run_name, optimizer=optimizer)


class FullContextNet(VocabCategoricalStandalone):
    def __init__(self, vocab_path: Path, batch_size: int,
                 params: Mapping[str, Any], out_dir: Path,
                 run_name: str = 'default',
                 optimizer=tf.keras.optimizers.Adam()):
        one_side = (tf.constant(value=""),) * params['ctx_len']
        dataset_producer = RegularCsvReader(one_side + (tf.string,) + one_side,
                                            (tf.string,),
                                            batch_size)
        tokens = [Input(shape=(), dtype=tf.string)
                  for i in range(params['ctx_len'] * 2 + 1)]
        core = ContextNet(params)
        super().__init__('contextnet', dataset_producer, vocab_path,
                         tokens, core(tokens), core,
                         out_dir, run_name, optimizer=optimizer)


class FullOccurenceNet(VocabCategoricalStandalone):
    def __init__(self, vocab_path: Path, batch_size: int,
                 params: Mapping[str, Any], out_dir: Path,
                 run_name: str = 'default',
                 optimizer=tf.keras.optimizers.Adam()):
        ctx_size = params['ctx_size']
        ctx_len = ctx_size * 2 + 1
        dataset_producer = OccurenceCsvReader(ctx_size, batch_size)
        inputs = (Input(shape=(None, ctx_len), dtype=tf.string),
                  Input(shape=(None, ctx_len), dtype=tf.int8))
        core = OccurenceNet(params)
        super().__init__('contextnet', dataset_producer, vocab_path,
                         inputs, core(inputs), core,
                         out_dir, run_name, optimizer=optimizer)
