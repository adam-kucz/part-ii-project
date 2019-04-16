from enum import auto, Enum, unique
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

from funcy import post_processing, lmap, partial, cached_property, lfilter

from .util import Interactive, cli_or_interactive
from ..util import csv_read

UNANNOTATED = '_'


class Record:
    identifier: str
    inputs: Tuple
    label: str
    predictions: List[Tuple[str, float]]

    def __init__(self, identifier, inputs, label, predictions):
        self.identifier = identifier
        self.inputs = inputs
        self.label = label
        self.predictions = predictions

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


@unique
class RecordMode(Enum):
    IDENTIFIER = auto()
    CONTEXT = auto()
    OCCURENCES = auto()


class Predictions(Interactive):
    records: List[Record]
    vocab: List[str]

    def __init__(self, dataset_path: Path, predictions_path: Path,
                 vocab_path: Optional[Path] = None,
                 mode: RecordMode = RecordMode.IDENTIFIER,
                 ctx_size: int = 0) -> None:
        self.vocab = lmap(0, csv_read(vocab_path)) if vocab_path else None
        self.mode = mode
        actual = csv_read(dataset_path)
        predictions = lmap(partial(lmap, eval), csv_read(predictions_path))
        assert len(actual) == len(predictions)

        def get_id(ctx):
            return ctx[ctx_size // 2]
        self.records = lmap(lambda t, p:
                            Record(get_id(t[:-1]), t[:-1], t[-1], p),
                            actual, predictions)

    @cached_property
    def correct(self) -> List[Record]:
        return lfilter(lambda r: r.correct(self.vocab), self.records)

    @cached_property
    def wrong(self) -> List[Record]:
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
