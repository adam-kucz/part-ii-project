"""Collection of useful methods that do not belong anywhere else"""
import csv
from hashlib import md5
from pathlib import Path
from typing import Any, Iterable, List, TypeVar

__all__ = ['csv_read', 'csv_write', 'stable_hash']


A = TypeVar('A')  # pylint: disable=invalid-name


def sortdict(data: A) -> A:
    if isinstance(data, dict):
        return dict(sorted(((k, sortdict(v)) for k, v in data.items()),
                           key=lambda t: t[0]))
    return data


def stable_hash(data: Any) -> bytes:
    """Returns a unique hash deterministic between runs"""
    encoded = repr(sortdict(data)).encode('utf-8')
    return md5(encoded).digest()  # nosec: B303


def csv_read(path: Path) -> List[List]:
    with path.open(newline='') as csvfile:
        return list(csv.reader(csvfile))


def csv_write(path: Path, rows: Iterable[List]):
    with path.open(mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(rows)
