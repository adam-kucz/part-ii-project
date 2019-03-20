from functools import partial
from pathlib import Path
from typing import Any, Optional

from ..generators.generator import Map
from ..generators.dlti import GenContext, TOKEN_CHARS
from ..generators.transform import Elements, Scale
from .primitive import Deterministic, InverseConditional


def reduce_class_number(example, classes,
                        new_num_classes: int, add_unk: bool = True):
    if add_unk and example not in classes:
        index = len(classes)
    else:
        index = classes.index(example)
    old_num_classes = len(classes) + (1 if add_unk else 0)
    if old_num_classes <= new_num_classes:
        return index
    return index * new_num_classes // old_num_classes


def nth_character_in_kth_ctx_element(
        nth: int, id_max_len: int, kth: int, ctx_len: int, num_classes: int,
        training_size: int, training_path: Path,
        validation_size: int, validation_path: Path,
        seed: Optional[Any] = None):
    def to_class(ctx):
        char = ctx[kth][nth] if len(ctx[kth]) > nth else ''
        return reduce_class_number(char, TOKEN_CHARS, num_classes)
    ctx_gen = Scale(lambda _: ctx_len, GenContext(id_max_len))
    scheme = Deterministic(to_class, ctx_gen, 0)
    scheme.save_datasets(training_size, training_path,
                         validation_size, validation_path, seed)


def length_of_kth_ctx_element(
        id_max_len: int, kth: int, ctx_len: int, num_classes: int,
        training_size: int, training_path: Path,
        validation_size: int, validation_path: Path,
        seed: Optional[Any] = None):
    def to_class(ctx):
        length = len(ctx[kth])
        return reduce_class_number(length, range(id_max_len + 1),
                                   num_classes, False)
    ctx_gen = Scale(lambda _: ctx_len, GenContext(id_max_len))
    scheme = Deterministic(to_class, ctx_gen, 0)
    scheme.save_datasets(training_size, training_path,
                         validation_size, validation_path, seed)


BEGINNINGS = ['int', 'def', '\n', 'float', 'str',
              'var', 'hi', 'step', 'cnt', 'num', 'fun', 'cls', 'self',
              'socket', 'user', 'acc', 'f', '__', '_']


def beginning_of_kth_ctx_element(
        id_max_len: int, kth: int, ctx_len: int, num_classes: int,
        training_size: int, training_path: Path,
        validation_size: int, validation_path: Path,
        seed: Optional[Any] = None):
    def add_to_kth(beginning, ctx):
        return ctx[:kth] + [beginning + ctx[kth]] + ctx[kth + 1:]
    ctx_gen = Scale(lambda _: ctx_len, GenContext(id_max_len))
    scheme = InverseConditional(lambda b: Map(partial(add_to_kth, b), ctx_gen),
                                Elements(BEGINNINGS[:num_classes]), 0)
    scheme.save_datasets(training_size, training_path,
                         validation_size, validation_path, seed)
