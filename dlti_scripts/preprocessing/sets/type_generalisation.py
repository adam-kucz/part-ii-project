from pathlib import Path
from typing import List, NamedTuple, Optional

from ..core.type_representation import Type
from ..util import csv_read, csv_write


def generalise_to_vocab(vocab: List[str], typ_str: str) -> str:
    def in_vocab(typ_str: str) -> Optional[str]:
        generalized = Type.from_str(typ_str)
        while generalized:
            if str(generalized) in vocab:
                return str(generalized)
            generalized = generalized.generalize()
        return None
    found = in_vocab(typ_str)
    return found if found else typ_str


class Generalisation(NamedTuple):
    prev_in_vocab: int
    now_in_vocab: int
    total: int


def generalise_file(vocab_filename: Path,
                    in_filename: Path, out_filename: Path) -> Generalisation:
    vocab = csv_read(vocab_filename)
    count, in_v, new_in_v = 0, 0, 0
    out_rows = []
    for row in csv_read(in_filename):
        count += 1
        general = generalise_to_vocab(vocab, row[-1])
        if row[-1] in vocab:
            in_v += 1
        if general in vocab:
            new_in_v += 1
        out_rows.append(row[:-1] + [general])
    csv_write(out_filename, out_rows)
    return Generalisation(in_v, new_in_v, count)
