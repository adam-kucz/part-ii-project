from pathlib import Path
from typing import List, Tuple, Iterable, Optional, NamedTuple

from funcy import (
    post_processing, lmap, partial, cached_property, lfilter, all, isa)

from .util import (Interactive, Record, RecordMode,
                   cli_or_interactive, read_dataset)
from ..util import csv_read

UNANNOTATED = '_'


class RecordWithPrediction(NamedTuple):
    identifier: str
    inputs: Tuple
    label: str
    predictions: List[Tuple[str, float]]

    @staticmethod
    def from_record(record: Record, predictions: List[Tuple[str, float]]):
        return RecordWithPrediction(*record, predictions)

    def most_likely(self) -> str:
        return self.predictions[0][0]

    def confidence(self):
        return 100 * self.predictions[0][1]

    @post_processing(any)
    def correct(self, vocab) -> Iterable[bool]:
        yield self.most_likely() == self.label
        yield self.most_likely() == UNANNOTATED and self.label not in vocab

    def top_k(self, k: int = 3) -> List[str]:
        return lmap(0, self.predictions[:k])

    def in_top_k(self, k: int = 3) -> bool:
        return self.label in self.top_k(k)


def get_full_records(mode, ctx_size, data_path: Path, predictions_path: Path)\
        -> Iterable[RecordWithPrediction]:
    predictions = map(partial(lmap, eval), csv_read(predictions_path))
    return map(RecordWithPrediction.from_record,
               read_dataset(mode, ctx_size, data_path),
               predictions)


class Predictions(Interactive):
    records: List[RecordWithPrediction]
    vocab: List[str]

    def __init__(self, dataset_path: Path, predictions_path: Path,
                 vocab_path: Optional[Path] = None,
                 mode: RecordMode = RecordMode.IDENTIFIER,
                 ctx_size: int = 0) -> None:
        self.vocab = lmap(0, csv_read(vocab_path)) if vocab_path else None
        self.mode = mode
        self.records = list(get_full_records(mode, ctx_size,
                                             dataset_path, predictions_path))

    @cached_property
    def correct(self) -> List[RecordWithPrediction]:
        result = lfilter(lambda r: r.correct(self.vocab), self.records)
        assert all(isa(RecordWithPrediction), result)
        return result

    @cached_property
    def wrong(self) -> List[RecordWithPrediction]:
        return lfilter(lambda r: not r.correct(self.vocab), self.records)


cli_or_interactive(
    Predictions,
    {("dataset_path",): {'type': Path},
     ('predictions_path',): {'type': Path},
     ('-v', '--vocab_path'): {'type': Path, 'default': None},
     ('-m', '--mode'): {'type': int,
                        'default': 0,
                        'help': ("Options are "
                                 "0 for 'identifier', "
                                 "1 for 'context', "
                                 "2 for 'occurence"),
                        'transform': {0: RecordMode.IDENTIFIER,
                                      1: RecordMode.CONTEXT,
                                      2: RecordMode.OCCURENCES}.__getitem__},
     ('-c', '--ctx_size'): {'type': int, 'default': 0}})
