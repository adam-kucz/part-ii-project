"""Collection of useful methods that do not belong anywhere else"""
from collections import defaultdict
import csv
from itertools import chain
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Mapping,
                    NamedTuple, Optional, Tuple, TypeVar)

from funcy import post_processing, suppress

from .core.syntactic.collectors import AccessError

PYTHON2_BUILTINS = {'StandardError', 'apply', 'basestring', 'buffer', 'cmp',
                    'coerce', 'execfile', 'file', 'intern', 'long',
                    'raw_input', 'reduce', 'reload', 'unichr',
                    'unicode', 'xrange'}

PYTHON2_STRING_PREFIXES = {"ur", "UR", "Ur", "uR"}


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
                err.args += (pypath,)
                raise
            exception_dict = (exceptions.python2 if _is_python2_error(err)
                              else exceptions.general)
            exception_dict[type(err)].append((pypath, err))
    return exceptions


@post_processing(any)
def _is_python2_error(error: Exception) -> Iterable[bool]:
    yield isinstance(error, SyntaxError)
    if isinstance(error, AccessError):
        name = error.name
        yield name.value in PYTHON2_BUILTINS
        next_leaf = name.get_next_leaf()
        yield (name.value in PYTHON2_STRING_PREFIXES
               and next_leaf.type == 'string')
        prev_leaf = name.get_previous_leaf()
        with suppress(AttributeError):
            try_block = name.parent.get_previous_sibling()\
                                   .get_previous_sibling()
            yield (next_leaf.value == ':' and prev_leaf.value == ','
                   and try_block[-1].type == 'except_clause')


def csv_read(path: Path) -> List[List]:
    with path.open(newline='') as csvfile:
        return list(csv.reader(csvfile))


def csv_write(path: Path, rows: Iterable[Iterable]):
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
