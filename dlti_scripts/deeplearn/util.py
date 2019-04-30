"""Collection of useful methods that do not belong anywhere else"""
import csv
from hashlib import md5
from inspect import signature
from pathlib import Path
from typing import Any, Iterable, List, TypeVar

from funcy import decorator

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


def augment_except(*args: str):
    @decorator
    def decorated(call):
        try:
            return call()
        except Exception as err:
            bound = signature(call._func).bind(*call._args, *call._kwargs)\
                                         .arguments
            err.args += tuple(bound[arg] for arg in args)
            raise
    return decorated


@augment_except('path')
def csv_read(path: Path) -> List[List]:
    with path.open(newline='') as csvfile:
        return list(csv.reader(csvfile))


@augment_except('path')
def csv_write(path: Path, rows: Iterable[List]):
    with path.open(mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(rows)
