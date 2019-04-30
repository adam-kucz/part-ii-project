from collections import defaultdict
from pathlib import Path
from typing import List, Mapping, Tuple, Iterable

from funcy import (map, lfilter, ilen, walk_values, cat,
                   cached_property, first, second)

from .predictions import RecordWithPrediction, get_full_records
from .util import RecordMode, summarise_numbers
from ..util import csv_read


class PredictionStats:
    mode: RecordMode
    ctx_size: int
    vocab: List[str]
    datasets: Mapping[str, List[RecordWithPrediction]]

    def __init__(self, paths: Iterable[Tuple[Path, Path]],
                 mode: RecordMode = RecordMode.IDENTIFIER,
                 ctx_size: int = 1,
                 vocab_path: Path = Path('vocab.csv'),
                 topk: Iterable[int] = (5,)):
        self.mode = mode
        self.ctx_size = ctx_size
        self.vocab = set(map(0, csv_read(vocab_path)))
        self.topk = list(topk)
        self.datasets = {
            data_path.stem: list(get_full_records(mode, ctx_size,
                                                  data_path, pred_path))
            for data_path, pred_path in paths}

    @cached_property
    def stats(self) -> Mapping[str, Mapping[str, float]]:
        if not self.datasets:
            return {}

        stats: Mapping[str, Mapping[str, float]] = defaultdict(dict)
        for name, data in self.datasets.items():
            # if len(data) < threshold:
            #     continue
            correct = lfilter(lambda r: r.correct(self.vocab), data)
            predictable = lfilter(lambda r: r.label in self.vocab, data)
            accurate = lfilter(lambda r: r.correct(self.vocab), predictable)
            covered = ilen(r for r in data if r.label in self.vocab)
            top_n = {n: (r for r in predictable
                         if r.label in map(0, r.predictions[:n]))
                     for n in self.topk}
            stats['size'][name] = len(data)
            stats['coverage'][name] = covered / len(data)
            stats['accuracy'][name] = len(correct) / len(data)
            if predictable:
                stats['real_accuracy'][name] = len(accurate) / len(predictable)
                for num in self.topk:
                    stats[f'real_top{num}'][name]\
                        = ilen(top_n[num]) / len(predictable)
        return stats

    def show_summary(self, threshold: int = 10):
        if not self.datasets:
            return

        summary_stats = walk_values(
            lambda mapping: summarise_numbers(
                n for name, n in mapping.items()
                if len(self.datasets[name]) > threshold),
            self.stats)
        namelen = max(len(name) for name in summary_stats)
        for stat, summary in summary_stats.items():
            average, std, *percentiles = summary
            percentiles = ', '.join(map(lambda p: f"{p:.2f}", percentiles))
            print(f"{stat:{namelen}}: "
                  f"{average:.2f} +- {std:.2f} [{percentiles}]")

    def show_stats(self):
        if not self.datasets:
            return

        ordered = self.stats.items()
        print('project_name', *map(first, ordered), sep=',')
        for proj in self.datasets:
            print(proj, *(mapping.get(proj, '') for _, mapping in ordered),
                  sep=',')
