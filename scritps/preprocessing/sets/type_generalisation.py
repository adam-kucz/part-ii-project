import csv
from pathlib import Path
from typing import List, NamedTuple, Optional

from ..core.type_representation import Type


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
    with in_filename.open(newline='') as infile,\
            out_filename.open(mode='w') as outfile:
        vocab = vocab_filename.read_text().split('\n')
        writer = csv.writer(outfile)
        count, in_v, new_in_v = 0, 0, 0
        for row in csv.reader(infile):
            count += 1
            general = generalise_to_vocab(vocab, row[-1])
            if row[-1] in vocab:
                in_v += 1
            if general in vocab:
                new_in_v += 1
            writer.writerow([row[:-1], general])
    return Generalisation(in_v, new_in_v, count)
