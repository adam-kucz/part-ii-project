from collections import Counter
import csv
from pathlib import Path
from typing import NamedTuple

from ..util import csv_read


class VocabStats(NamedTuple):
    included: int
    total: int


def create_vocab(in_filename: Path,
                 out_filename: Path, percentage: float) -> VocabStats:
    with out_filename.open(mode='w', newline='') as vocab_file:
        types = [row[-1] for row in csv_read(in_filename)]
        writer = csv.writer(vocab_file)
        vocab_size = 0
        included = 0
        for typ, count in Counter(types).most_common():
            writer.writerow([typ])
            vocab_size = 0
            included += count
            if included / len(types) >= percentage:
                return VocabStats(vocab_size, len(Counter(types)))
    raise ValueError(in_filename, percentage)
