#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
import traceback as tb
from typing import Any, Dict, Optional

from preprocessing.core.context.extract import extract_type_contexts
from preprocessing.core.pairs.extract import extract_type_identifiers
from preprocessing.core.syntactic.extract import extract_all_syntactic_contexts
from preprocessing.util import extract_dir

MODES = {'identifier': (extract_type_identifiers,
                        {'f': 'func_as_ret'}),
         'context': (extract_type_contexts,
                     {'ctx_size': 'context_size', 'f': 'func_as_ret'}),
         'syntax': (extract_all_syntactic_contexts,
                    {'ctx_size': 'context_size', 'f': 'func_as_ret'})}


def main(inpath: Path, outpath: Path, mode: str,
         kwarg_dict: Dict[str, Any],
         logdir: Optional[Path] = None, fail_fast: bool = False):
    function, arglist = MODES[mode]
    kwargs = dict((arglist[k], kwarg_dict[k]) for k in arglist)

    exceptions = extract_dir(inpath, outpath,
                             partial(function, **kwargs), fail_fast)
    if logdir:
        if not logdir.exists():
            logdir.mkdir(parents=True)
        for exception_dict, prefix in zip(exceptions, ("", "Python2-")):
            for exception_type, instances in exception_dict.items():
                filename = prefix + exception_type.__name__
                with logdir.joinpath(filename).open('w') as fileout:
                    for path, exception in instances:
                        tb.print_exception(exception_type, exception,
                                           exception.__traceback__,
                                           file=fileout)
                        print("\nSource: {}\n\n".format(path), file=fileout)
    else:
        for exception_dict, suffix in zip(exceptions, ("", " [Python2]")):
            for exception_type, instances in exception_dict.items():
                print("Exceptions {}:"
                      .format(exception_type.__name__ + suffix))
                for path, exception in instances:
                    tb.print_exception(exception_type, exception,
                                       exception.__traceback__)
                    print("\nSource: {}\n\n".format(path), file=fileout)


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
    PARSER.add_argument('-q', '--fail_fast', action='store_true',
                        help="fail at first exception")
    ARGS: Namespace = PARSER.parse_args()

    OUTDIR: Path = (ARGS.outdir
                    or ARGS.repodir.parent.joinpath('data', 'sets', ARGS.mode))
    main(ARGS.repodir, OUTDIR, ARGS.mode, vars(ARGS),
         ARGS.logdir, fail_fast=ARGS.fail_fast)
