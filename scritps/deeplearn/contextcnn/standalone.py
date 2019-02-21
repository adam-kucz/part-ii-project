from pathlib import Path
from typing import Any, Callable, List, Mapping

import tensorflow as tf

from .contextcnn import ContextCNN
from ..data_ops.data_interface import CsvReader, PairInterface
from ..data_ops.data_transformers import (
    Categorical, HeadTailProc, InitLastProc, ProcTrans, StrEnc)
from ..charcnn.network_assembly import Modular
from ..charcnn.trainer import Trainer
from ..charcnn.typeclass_output import TypeclassOutput

__all__ = ['FullContextCNN']


class FullContextCNN(Trainer):
    def __init__(self,
                 vocab_path: Path,
                 identifier_len: int,
                 ctx_len: int,
                 ctx_size: int,
                 batch_size: int,
                 params: Mapping[str, Any],
                 out_dir: Path,
                 log: Callable[[str], None] = lambda _: None):
        vocab: List[str] = vocab_path.read_text().splitlines()
        dataset_producer = CsvReader((tf.string,) * ctx_size,
                                     (tf.string,),
                                     batch_size)
        context_proc = HeadTailProc(StrEnc(identifier_len), StrEnc(ctx_len))
        data_processor = InitLastProc(ProcTrans(context_proc),
                                      Categorical(vocab, unk=True))
        core_net = ContextCNN(params, log)
        output_net = TypeclassOutput(len(vocab) + 1, log)
        super().__init__(dataset_producer,
                         PairInterface(),
                         Modular(data_processor, core_net, output_net, log),
                         out_dir)
