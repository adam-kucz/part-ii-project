from pathlib import Path
from typing import Any, Optional

from ..generators.dlti import GenContext, GEN_IDENTIFIER
from ..generators.transform import Scale
from .primitive import Deterministic


def nth_character_identifier(nth: int, id_max_len: int,
                             training_size: int, training_path: Path,
                             validation_size: int, validation_path: Path,
                             seed: Optional[Any] = None):
    scheme = Deterministic(lambda x: x[nth] if len(x) > nth else '',
                           GEN_IDENTIFIER, id_max_len)
    scheme.save_datasets(training_size, training_path,
                         validation_size, validation_path, seed)


def length_of_identifier(id_max_len: int,
                         training_size: int, training_path: Path,
                         validation_size: int, validation_path: Path,
                         seed: Optional[Any] = None):
    scheme = Deterministic(len, GEN_IDENTIFIER, id_max_len)
    scheme.save_datasets(training_size, training_path,
                         validation_size, validation_path, seed)


def nth_character_in_kth_ctx_element(
        nth: int, id_max_len: int, kth: int, ctx_len: int,
        training_size: int, training_path: Path,
        validation_size: int, validation_path: Path,
        seed: Optional[Any] = None):
    def get_the_char(ctx):
        return ctx[kth][nth] if len(ctx[kth]) > nth else ''
    ctx_gen = Scale(lambda _: ctx_len, GenContext(id_max_len))
    scheme = Deterministic(get_the_char, ctx_gen, 0)
    scheme.save_datasets(training_size, training_path,
                         validation_size, validation_path, seed)
