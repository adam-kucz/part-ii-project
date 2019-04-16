#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from pathlib import Path
from pprint import pprint
import sys
import traceback as tb
from typing import Any, Dict, Optional, Tuple, TextIO

from funcy import walk_values, print_durations, walk_keys, compose

from preprocessing.core.type_collector import TypeCollector
from preprocessing.core.context.extract import (
    extract_type_contexts, get_context)
from preprocessing.core.cst_util import ScopeAwareNodeVisitor, pure_annotation
from preprocessing.core.pairs.extract import extract_type_identifiers
from preprocessing.core.syntactic.collectors import BindingCollector
from preprocessing.core.syntactic.extract import (
    extract_all_syntactic_contexts, attraccess, import_src, arg_keyword)
from preprocessing.util import extract_dir, track_total_time

MODES = {'identifier': (extract_type_identifiers,
                        {'f': 'func_as_ret'},
                        (TypeCollector,)),
         'context': (extract_type_contexts,
                     {'ctx_size': 'context_size',
                      'f': 'func_as_ret',
                      'c': 'include_comments'},
                     (TypeCollector, pure_annotation, get_context)),
         'occurence': (extract_all_syntactic_contexts,
                       {'ctx_size': 'context_size',
                        'f': 'func_as_ret',
                        'c': 'include_comments'},
                       (TypeCollector, BindingCollector, ScopeAwareNodeVisitor,
                        pure_annotation, attraccess, import_src, arg_keyword,
                        get_context))}


def print_extra_logs_to(extra_logs: Tuple, fileout: TextIO):
    tmp = sys.stdout
    sys.stdout = fileout
    print(f"Valid files: {extract_dir.count}")
    for tracker in track_total_time.trackers:
        print(f"Total time spent in {tracker.__name__}: {tracker.total_time}")
    for logtype in extra_logs:
        if logtype is TypeCollector:
            print("Function type comments ignored: "
                  f"{logtype.other_logs['func_type_comments']}")
            print("Types collected")
            pprint(walk_values(len, dict(logtype.type_logs)))
        elif logtype is pure_annotation:
            print(f"Unassigned annassign found: {len(logtype.names)}")
            pprint(set(map(lambda n: n.value, logtype.names)))
        elif logtype is BindingCollector:
            print("Bindings")
            pprint(walk_values(len, dict(logtype.binding_logs)))
        elif logtype is ScopeAwareNodeVisitor:
            print("Scopes")
            pprint(walk_keys(compose(tuple, partial(map, str)),
                             dict(logtype.scope_logs)))
        elif logtype is get_context:
            print(f"Comments before: {logtype.count_before}")
            print(f"Comments after: {logtype.count_after}")
        elif logtype in (attraccess, import_src, arg_keyword):
            print(f"{logtype.__name__.capitalize()} found: {logtype.count}")
    sys.stdout = tmp


@print_durations(repr_len=50)
def main(inpath: Path, outpath: Path, mode: str,
         kwarg_dict: Dict[str, Any],
         logdir: Optional[Path] = None, fail_fast: bool = False):
    function, arglist, extra_logs = MODES[mode]
    kwargs = dict((arglist[k], kwarg_dict[k]) for k in arglist)

    exceptions = extract_dir(inpath, outpath,
                             partial(function, **kwargs), fail_fast)
    exception_count: Dict[str, int] = defaultdict(int)
    if logdir:
        if not logdir.exists():
            logdir.mkdir(parents=True)
        for err_type, version, err, err_path in exceptions:
            prefix = "Python{}.{}-".format(*version) if version else ""
            exception_name = prefix + err_type.__name__
            path = logdir.joinpath(exception_name)
            if exception_name not in exception_count and path.exists():
                path.unlink()
            exception_count[exception_name] += 1
            with path.open('a') as fileout:
                if not version:
                    tb.print_exception(err_type, err,
                                       err.__traceback__, file=fileout)
                    print("\nSource: {}\n\n".format(err_path), file=fileout)
                else:
                    print(err, file=fileout)
                    print("In {}".format(err_path), file=fileout)
        with logdir.joinpath("extra_logs.txt").open('w') as fout:
            print_extra_logs_to(extra_logs, fout)
    else:
        for err_type, version, err, err_path in exceptions:
            prefix = "Python{}.{}-".format(*version) if version else ""
            exception_name = prefix + err_type.__name__
            exception_count[exception_name] += 1
            print("Exception {}:".format(exception_name))
            if not version:
                tb.print_exception(err_type, err, err.__traceback__)
                print("\nSource: {}\n\n".format(err_path))
            else:
                print(err)
                print("In {}".format(err_path))
        print_extra_logs_to(extra_logs, sys.stdout)
    pprint(exception_count)


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description="Extract (name, type) pairs from python source file")
    PARSER.add_argument('mode', type=str,
                        help="one of 'identifier', 'context'")
    PARSER.add_argument('repodir', type=Path,
                        help="directory with all repositories")
    PARSER.add_argument('outdir', type=Path, nargs='?', default=None,
                        help="output directory for extracted types")
    PARSER.add_argument('-l', '--logdir', type=Path, nargs='?', default=None,
                        help="log directory for exceptions raised")
    PARSER.add_argument('-s', '--ctx_size', type=int, nargs='?', default=5,
                        help="number of tokens in the context (on each side)"
                        "only applicable for 'context' mode")
    PARSER.add_argument('-f', action='store_true',
                        help="assign return types to function identifiers"
                        " instead of Callable[[...], ...]")
    PARSER.add_argument('-c', action='store_true',
                        help="include non-type comments as part of context")
    PARSER.add_argument('-q', '--fail_fast', action='store_true',
                        help="fail at first exception")
    ARGS: Namespace = PARSER.parse_args()

    OUTDIR: Path = (ARGS.outdir
                    or ARGS.repodir.parent.joinpath('data', 'sets', ARGS.mode))
    main(ARGS.repodir, OUTDIR, ARGS.mode, vars(ARGS),
         ARGS.logdir, fail_fast=ARGS.fail_fast)
