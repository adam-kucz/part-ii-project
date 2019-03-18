import csv
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar

from ..generators.generator import Generator

__all__ = ['Scheme']

T = TypeVar('T')


class Scheme(Generic[T]):
    def __init__(self, generator: Generator[T], size: int):
        self.generator: Generator[T] = generator
        self.size: int = size

    def get_examples(self, num: int, seed: Optional[Any] = None) -> List[T]:
        if seed:
            Generator.set_seed(seed)
        return [self.generator.generate(self.size) for _ in range(num)]

    def save_csv(self, num: int, filename: Path, seed: Optional[Any] = None):
        with filename.open('w', newline='') as csvfile:
            csv.writer(csvfile).writerows(self.get_examples(num, seed))

    def save_datasets(self, training_size: int, training_path: Path,
                      validation_size: int, validation_path: Path,
                      seed: Optional[Any] = None):
        self.save_csv(training_size, training_path, seed)
        self.save_csv(validation_size, validation_path)
