"""Collection of useful methods that do not belong anywhere else"""
from collections import defaultdict
import csv
from itertools import chain
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Mapping,
                    NamedTuple, Optional, Tuple, TypeVar)

import preprocessing.core as core

PYTHON2_BUILTINS = ['StandardError', 'apply', 'basestring', 'buffer', 'cmp',
                    'coerce', 'execfile', 'file', 'intern', 'long',
                    'raw_input', 'reduce', 'reload', 'unichr',
                    'unicode', 'xrange']


A = TypeVar('A')  # pylint: disable=invalid-name
B = TypeVar('B')  # pylint: disable=invalid-name
T = TypeVar('T')  # pylint: disable=invalid-name


# functional conventions use short names because objects are very abstract
# pylint: disable=invalid-name
def bind(a: Optional[A], f: Callable[[A], Optional[B]]) -> Optional[B]:
    """Monadic bind for the Option monad"""
    return f(a) if a else None


class GatheredExceptions(NamedTuple):
    general: Dict[type, List[Tuple[Path, Exception]]]
    python2: Dict[type, List[Tuple[Path, Exception]]]

    @property
    def all(self) -> Mapping[type, List[Tuple[Path, Exception]]]:
        return dict(chain(self.general.items(), self.python2.items()))


def extract_dir(repo_dir: Path, out_dir: Path,
                extraction_function: Callable[[Path, Path], None],
                fail_fast: bool = False, ignore_python2: bool = True)\
                -> GatheredExceptions:
    """
    Extracts annotaions from all files in the directory

    Stores the files in per-repo subdirectories of out_dir
    """
    exceptions = GatheredExceptions(defaultdict(list), defaultdict(list))
    for pypath in filter(lambda p: p.is_file(), repo_dir.rglob('*.py')):\
            # type: Path
        rel: Path = pypath.relative_to(repo_dir)
        repo: str = rel.parts[0]
        outpath: Path = out_dir.joinpath(repo, '+'.join(rel.parts[1:]))\
                               .with_suffix('.csv')
        try:
            extraction_function(pypath, outpath)
        # we want to catch and report *all* exceptions
        except Exception as err:  # pylint: disable=broad-except
            if fail_fast and not (ignore_python2 and _is_python2_error(err)):
                err.args += ("File {}".format(pypath),)
                raise
            exception_dict = (exceptions.python2 if _is_python2_error(err)
                              else exceptions.general)
            exception_dict[type(err)].append((pypath, err))
    return exceptions


def _is_python2_error(error: Exception):
    return ((isinstance(error, core.ast_util.AccessError)
             and error.name in PYTHON2_BUILTINS)
            or isinstance(error, SyntaxError))


def csv_read(path: Path) -> List[List]:
    with path.open(newline='') as csvfile:
        return list(csv.reader(csvfile))


def csv_write(path: Path, rows: Iterable[List]):
    with path.open(mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(rows)


def intersperse(sep: T, iterable: Iterable[T]) -> Iterable[T]:
    it = iter(iterable)
    try:
        yield next(it)
    except StopIteration:
        return
    for x in it:
        yield sep
        yield x
