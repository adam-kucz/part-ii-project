from argparse import ArgumentParser, Namespace
from pathlib import Path

from deeplearn.preprocessing.extract_type_and_context import (
    extract_type_contexts)


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Extract (ctx, type) pairs from python source file')
    PARSER.add_argument('path', type=Path, help='source file')
    PARSER.add_argument('out', type=Path, nargs='?', default=Path('out.csv'),
                        help='output file')
    PARSER.add_argument(
        'ctx_size', type=int, nargs='?', default=5,
        help='size of context (number of elements on each side)')
    PARSER.add_argument(
        '-f', action='store_true',
        help='assign return types to function identifiers'
        ' instead of Callable[[...], ...]')
    ARGS: Namespace = PARSER.parse_args()

    extract_type_contexts(ARGS.path, ARGS.out, ARGS.ctx_size, ARGS.f)
