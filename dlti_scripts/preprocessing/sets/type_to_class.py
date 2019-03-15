from collections import Counter
import csv
from pathlib import Path
from typing import NamedTuple


class VocabStats(NamedTuple):
    included: int
    total: int


def create_vocab(in_filename: Path,
                 out_filename: Path, percentage: float) -> VocabStats:
    with in_filename.open(newline='') as csvfile,\
         out_filename.open(mode='w') as vocabfile:
        reader = csv.reader(csvfile, delimiter=',')
        types = tuple(map(lambda t: t[-1], reader))
        vocab_size = 0
        included = 0
        for typ, count in Counter(types).most_common():
            print(typ, file=vocabfile)
            vocab_size = 0
            included += count
            if included / len(types) >= percentage:
                return VocabStats(vocab_size, len(Counter(types)))
