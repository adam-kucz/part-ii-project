#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

from preprocessing.core.context.extract_type_and_context import (
    extract_type_contexts)
from preprocessing.core.pairs.extract_type_pairs import (
    extract_type_annotations)
from preprocessing.util import extract_dir


def main(inpath: Path, outpath: Path, mode: str, kwarg_dict):
    if mode == 'identifier':
        function = extract_type_annotations
        arglist = {'f': 'fun_as_ret'}
    elif ARGS.mode == 'context':
        function = extract_type_contexts
        arglist = {'ctx_size': 'context_size', 'f': 'fun_as_ret'}
    else:
        raise ValueError("Unrecognised mode: {}".format(mode))
    kwargs = dict((arglist[k], kwarg_dict[k]) for k in arglist)

    exceptions = extract_dir(inpath, outpath, partial(function, **kwargs))
    if exceptions:
        print("Exceptions:")
        for path, exception in exceptions:
            print("Exception {}\nSource: {}".format(exception, path))


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description="Extract (name, type) pairs from python source file")
    PARSER.add_argument('mode', type=str,
                        help="one of 'identifier', 'context'")
    PARSER.add_argument('repodir', type=Path,
                        help="directory with all repositories")
    PARSER.add_argument('outdir', type=Path, nargs='?', default=None,
                        help="output directory for extracted types")
    PARSER.add_argument('-s', '--ctx_size', type=int, nargs='?', default=5,
                        help="number of tokens in the context (on each side)"
                        "only applicable for 'context' mode")
    PARSER.add_argument('-f', action='store_true',
                        help="assign return types to function identifiers"
                        " instead of Callable[[...], ...]")
    ARGS: Namespace = PARSER.parse_args()

    OUTDIR: Path = (ARGS.outdir
                    or ARGS.repodir.parent.joinpath('data', 'sets', ARGS.mode))
    main(ARGS.repodir, OUTDIR, ARGS.mode, vars(ARGS))
