#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from preprocessing.core.context.extract import extract_type_contexts
from preprocessing.core.pairs.extract import extract_type_identifiers
from preprocessing.util import extract_dir

MODES = {'identifier': (extract_type_identifiers,
                        {'f': 'fun_as_ret'}),
         'context': (extract_type_contexts,
                     {'ctx_size': 'context_size', 'f': 'fun_as_ret'})}


def main(inpath: Path, outpath: Path, mode: str,
         kwarg_dict: Dict[str, Any], logdir: Optional[Path] = None):
    function, arglist = MODES[mode]
    kwargs = dict((arglist[k], kwarg_dict[k]) for k in arglist)

    exceptions = extract_dir(inpath, outpath, partial(function, **kwargs))
    if logdir and not logdir.exists():
        logdir.mkdir(parents=True)
    for exception_type in exceptions:
        if logdir:
            with logdir.joinpath(str(exception_type)).open('w') as fileout:
                for path, exception in exceptions[exception_type]:
                    fileout.write("{}, {}".format(exception, path))
        else:
            print("Exceptions {}:".format(exception_type))
            for path, exception in exceptions[exception_type]:
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
