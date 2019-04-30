"""Collection of useful methods that do not belong anywhere else"""
import csv
from inspect import signature
from pathlib import Path
import sys
from time import perf_counter as timer
from typing import (Callable, Iterable, List, NamedTuple, Optional, TypeVar)

from funcy import post_processing, some, decorator
import parso
from parso.utils import PythonVersionInfo

PYTHON2_BUILTINS = {'StandardError', 'apply', 'basestring', 'buffer', 'cmp',
                    'coerce', 'execfile', 'file', 'intern', 'long',
                    'raw_input', 'reduce', 'reload', 'unichr',
                    'unicode', 'xrange'}


A = TypeVar('A')  # pylint: disable=invalid-name
B = TypeVar('B')  # pylint: disable=invalid-name
T = TypeVar('T')  # pylint: disable=invalid-name


# functional conventions use short names because objects are very abstract
# pylint: disable=invalid-name
def bind(a: Optional[A], f: Callable[[A], Optional[B]]) -> Optional[B]:
    """Monadic bind for the Option monad"""
    return f(a) if a else None


def static_vars(**kwargs):
    def decorate(func):
        for key, value in kwargs.items():
            if hasattr(func, key):
                raise ValueError("Function already has attribute", func, key)
            setattr(func, key, value)
        return func
    return decorate


class AccessError(NameError):
    def __init__(self, name: parso.python.tree.Name):
        self.name = name
        super().__init__("Unbound name", self.name)


class ExceptionRecord(NamedTuple):
    type: type
    version: Optional[PythonVersionInfo]
    exception: Exception
    path: Path


@decorator
def _total_time_wrapper(call):
    func = call._func
    if func.nesting == 0:
        func.nesting += 1
        start = timer()
        value = call()
        func.total_time += timer() - start
        func.nesting -= 1
        return value
    func.nesting += 1
    value = call()
    func.nesting -= 1
    return value


@static_vars(trackers=set())
def track_total_time(func):
    func.nesting = 0
    func.total_time = 0
    track_total_time.trackers.add(func)
    return _total_time_wrapper(func)


def format_time(sec):
    if sec < 1e-6:
        mul, unit = 1e9, 'ns'
    elif sec < 1e-3:
        mul, unit = 1e6, 'us'
    elif sec < 1:
        mul, unit = 1e3, 'ms'
    elif sec < 100:
        mul, unit = 1, 's'
    elif sec < 3600:
        mul, unit = 1 / 60, 'min'
    else:
        mul, unit = 1 / 3600, 'h'
    return f"{sec * mul:8.2f} {unit:>3}"


def with_durations(seq, prefix="", label=""):
    """Prints time of processing of each item in seq.

    Like in funcy, but label can be a function of item"""
    it = iter(seq)
    for i, item in enumerate(it):
        print(prefix.format(i=i, item=item) if type(prefix) == str
              else prefix(i=i, item=item),
              end='')
        sys.stdout.flush()
        start = timer()
        yield item
        print(f"{format_time(timer() - start)} in iteration {i}"
              + (label.format(i=i, item=item) if type(label) == str
                 else label(i=i, item=item)))


@track_total_time
@static_vars(count=0)
def extract_dir(repo_dir: Path, out_dir: Path,
                extraction_function: Callable[[Path, Path], None],
                fail_fast: bool = False, ignore_python2: bool = True)\
                -> Iterable[ExceptionRecord]:
    """
    Extracts annotaions from all files in the directory

    Stores the files in per-repo subdirectories of out_dir
    """
    for projdir in with_durations(
            filter(Path.is_dir, repo_dir.iterdir()),
            prefix=lambda i, item: f"Extracting from {str(item) + ' ...':35}"):
        for pypath in filter(lambda p: p.is_file(), projdir.rglob('*.py')):\
                # type: Path
            rel: Path = pypath.relative_to(repo_dir)
            repo: str = rel.parts[0]
            outpath: Path = out_dir.joinpath(repo, '+'.join(rel.parts[1:]))\
                                   .with_suffix('.csv')
            try:
                extraction_function(pypath, outpath)
                extract_dir.count += 1
            # we want to catch and report *all* exceptions
            except Exception as err:  # pylint: disable=broad-except
                if fail_fast and not (ignore_python2
                                      and _is_python2_error(err)):
                    err.args += (pypath,)
                    raise
                version = (PythonVersionInfo(2, 7) if _is_python2_error(err)
                           else None)
                yield ExceptionRecord(type(err), version, err, pypath)


@post_processing(any)
def _is_python2_error(error: Exception) -> Iterable[bool]:
    # TODO: add node parameter to these errors and check for parsing in 2.7
    yield isinstance(error, (SyntaxError, UnicodeDecodeError))
    if isinstance(error, AccessError):
        name = error.name
        yield name.value in PYTHON2_BUILTINS
        code = name.get_root_node().get_code()
        grammar = parso.load_grammar(version='2.7')
        yield some(grammar.iter_errors(grammar.parse(code))) is None
        # yield (name.value in PYTHON2_STRING_PREFIXES
        #        and next_leaf.type == 'string')
        # prev_leaf = name.get_previous_leaf()
        # with suppress(AttributeError):
        #     try_block = name.parent.get_previous_sibling()\
        #                            .get_previous_sibling()
        #     yield (next_leaf.value == ':' and prev_leaf.value == ','
        #            and try_block[-1].type == 'except_clause')


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


@track_total_time
@augment_except('path')
def csv_read(path: Path) -> List[List]:
    with path.open(newline='') as csvfile:
        return list(csv.reader(csvfile))


@track_total_time
@augment_except('path')
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
