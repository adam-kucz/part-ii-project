from pathlib import Path
from typing import Container, NamedTuple

from funcy import lmap

from ..core.type_representation import Type
from ..util import csv_read, csv_write, augment_except


@augment_except('typ_str')
def generalise_to_vocab(vocab: Container[str], typ_str: str) -> str:
    generalized = Type.from_str(typ_str)
    while generalized:
        if str(generalized) in vocab:
            return str(generalized)
        generalized = generalized.generalize()
    return typ_str


class Generalisation(NamedTuple):
    prev_in_vocab: int
    now_in_vocab: int
    total: int


@augment_except('in_filename')
def generalise_file(
        vocab_filename: Path, in_filename: Path, out_filename: Path,
        verbose: int = 1) -> Generalisation:
    vocab = lmap(0, csv_read(vocab_filename))
    count, in_v, new_in_v = 0, 0, 0
    out_rows = []
    for row in csv_read(in_filename):
        count += 1
        general = generalise_to_vocab(vocab, row[-1])
        if row[-1] in vocab:
            in_v += 1
        if general in vocab:
            new_in_v += 1
        if verbose and general != row[-1]:
            print(f"Generalised {row[-1]} to {general}")
        out_rows.append(row[:-1] + [general])
    csv_write(out_filename, out_rows)
    return Generalisation(in_v, new_in_v, count)
