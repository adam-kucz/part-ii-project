from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, List, Callable, Set

from funcy import ilen, cat, map, walk_values, re_test

from ..util import csv_read
from .util import (
    Interactive, Record, RecordMode,
    pickable_option, cli_or_interactive, read_dataset, summarise_numbers)


def default_criterion(path: Path, vocab: Path, train: Path) -> bool:
    return (path.is_file()
            and path.resolve() != vocab.resolve()
            and not path.stem.endswith('original')
            and not path.stem.endswith('general'))


def _format_rel_amt(type_count, total_type_count, ndigits=5):
    return (f"{type_count:{ndigits}} "
            f"({100 * type_count / total_type_count:6.2f}%)")


class Datasets(Interactive):
    mode: RecordMode
    ctx_size: int
    vocab: Set[str]
    datasets: Mapping[str, List[Record]]

    def __init__(self, dataset_folder: Path,
                 mode: RecordMode = RecordMode.IDENTIFIER,
                 ctx_size: int = 1,
                 train_path: Path = Path('train.csv'),
                 vocab_path: Path = Path('vocab.csv'),
                 criterion: Callable[[Path], bool] = default_criterion):
        self.mode = mode
        self.ctx_size = ctx_size
        self.vocab = set(map(0, csv_read(vocab_path)))
        self.train_dataset = read_dataset(mode, ctx_size, train_path)
        self.datasets = {path.stem: read_dataset(mode, ctx_size, path)
                         for path in dataset_folder.glob("*.csv")
                         if criterion(path, vocab_path, train_path)}

    @pickable_option
    def summary_stats(self):
        stats: Mapping[str, Mapping[str, float]] = defaultdict(dict)
        for name, data in self.datasets.items():
            if len(data) < 10:
                continue
            dataset_types = Counter(r.label for r in data)
            covered = sum(n for i, n in dataset_types.items()
                          if i in self.vocab)
            other_types = set(cat((r.label for r in o_data)
                                  for o_name, o_data in self.datasets.items()
                                  if o_name != name))
            num_specific_records = sum(n for t, n in dataset_types.items()
                                       if t not in other_types)
            stats['size'][name] = len(data)
            stats['coverage'][name] = covered / len(data)
            stats['unique'][name] = num_specific_records / len(data)
        # TODO: names hardcoded, maybe fix later
        summary_stats = walk_values(
            lambda mapping: summarise_numbers(n for name, n in mapping.items()
                                              if not re_test('train', name)),
            stats)
        for stat, summary in summary_stats.items():
            average, std, *percentiles = summary
            percentiles = ', '.join(map(lambda p: f"{p:.2f}", percentiles))
            print(f"{stat}: {average:.2f} +- {std:.2f} [{percentiles}]")

    @pickable_option
    def individual_stats(self):
        print("Datasets statistics:")
        records = self.datasets.values()
        record_num = ilen(cat(records))
        digit_num = 7
        print(f"Total records: {record_num}")
        all_types = Counter(r.label for r in cat(records))
        print(f"Total unique types: {len(all_types)}")
        print(f"Vocabulary size: {len(self.vocab)}")
        coverage = sum(n for i, n in all_types.items() if i in self.vocab)
        print(f"Vocabulary coverage: {_format_rel_amt(coverage, record_num)}")
        print("Columns: #items, #ids, #types, #unique types, "
              ", #items w/ unique types, vocab coverage")
        for name, data in sorted(self.datasets.items()):
            print(f"Dataset '{name}'")
            item_num = len(data)
            dataset_types = Counter(r.label for r in data)
            other_types = set(cat((r.label for r in o_data)
                                  for o_name, o_data in self.datasets.items()
                                  if o_name != name))
            num_specific_types = ilen(t for t in dataset_types
                                      if t not in other_types)
            # if len(dataset_types) < 20 and num_specific_types == 0:
            #     for typ in dataset_types:
            #         other = some(o_name
            #                      for o_name, o_data in self.datasets.items()
            #                      if (o_name != name
            #                          and typ in (r.label for r in o_data)))
            #         print(f"Type {typ} is also found in {other}")
            num_specific_records = sum(n for t, n in dataset_types.items()
                                       if t not in other_types)
            coverage = sum(n for i, n in dataset_types.items()
                           if i in self.vocab)
            print(f"{item_num:{digit_num}}",
                  f"{len(set(map(lambda r: r.identifier, data))):{digit_num}}",
                  f"{_format_rel_amt(len(dataset_types), len(all_types))}",
                  f"{_format_rel_amt(num_specific_types, len(dataset_types))}",
                  f"{_format_rel_amt(num_specific_records, len(data))}",
                  f"{_format_rel_amt(coverage, len(data))}",
                  sep=', ')


cli_or_interactive(
    Datasets,
    {("dataset_folder",): {'type': Path},
     ('-m', '--mode'): {'type': int,
                        'default': 0,
                        'help': ("Options are "
                                 "0 for 'identifier', "
                                 "1 for 'context', "
                                 "2 for 'occurence"),
                        'transform': {0: RecordMode.IDENTIFIER,
                                      1: RecordMode.CONTEXT,
                                      2: RecordMode.OCCURENCES}.__getitem__},
     ('-c', '--ctx_size'): {'type': int, 'default': 1,
                            'help': ("Only needed for 'occurence' mode, "
                                     "otherwise deduced from datasets")},
     ('-v', '--vocab_path'): {'type': Path, 'default': Path('vocab.csv'),
                              'help': "Path to the vocabulary file"},
     ('-t', '--train_path'): {'type': Path, 'default': Path('train.csv'),
                              'help': "Path to the train dataset"}})
