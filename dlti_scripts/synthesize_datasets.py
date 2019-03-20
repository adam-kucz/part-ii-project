from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict

import datagen.schemes.particular as generate

UNIVERSAL = {'id_max_len': 'id_max_len',
             'num_classes': 'num_classes',
             'ctx_len': 'ctx_len'}
MODES = {'id-char': (generate.nth_character_in_kth_ctx_element,
                     dict(nth='token_char', kth='ctx_len', **UNIVERSAL)),
         'id-len': (generate.length_of_kth_ctx_element,
                    dict(kth='ctx_len', **UNIVERSAL)),
         'id-beg': (generate.beginning_of_kth_ctx_element,
                    dict(kth='ctx_len', **UNIVERSAL)),
         'ctx-tok-char': (generate.nth_character_in_kth_ctx_element,
                          dict(nth='token_char', kth='ctx_elem', **UNIVERSAL)),
         'ctx-tok-len': (generate.length_of_kth_ctx_element,
                         dict(kth='ctx_elem', **UNIVERSAL)),
         'ctx-tok-beg': (generate.beginning_of_kth_ctx_element,
                         dict(kth='ctx_elem', **UNIVERSAL))}


def main(datadir: Path, mode: str,
         training_size: int, validation_size: int,
         kwarg_dict: Dict[str, Any]):
    function, arglist = MODES[mode]
    kwargs = dict((k, kwarg_dict[arglist[k]]) for k in arglist)

    if not datadir.exists():
        datadir.mkdir(parents=True)

    function(training_size=training_size,
             training_path=datadir.joinpath("train.csv"),
             validation_size=validation_size,
             validation_path=datadir.joinpath("validate.csv"),
             **kwargs)


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description="Synthesize data for testing",
        formatter_class=ArgumentDefaultsHelpFormatter)
    PARSER.add_argument('mode', type=str,
                        help=("currently supported '{}'"
                              .format("', '".join(MODES.keys()))))
    PARSER.add_argument('datadir', type=Path,
                        help="directory to save the data to")
    PARSER.add_argument('-t', '--train_size', type=int, default=40000,
                        help="number of train set examples to generate")
    PARSER.add_argument('-v', '--validation_size', type=int, default=20000,
                        help="number of validation set examples to generate")
    PARSER.add_argument('-n', '--num_classes', type=int, default=16,
                        help="number of classes to split examples into")
    PARSER.add_argument('-i', '--token_char', type=int, default=1,
                        help="which character of token should decide type")
    PARSER.add_argument('-l', '--id_max_len', type=int, default=20,
                        help="maximum length of identifier")
    PARSER.add_argument('-k', '--ctx_elem', type=int, default=0,
                        help=("which element of context (including identifier)"
                              " should decide type"))
    PARSER.add_argument('-c', '--ctx_len', type=int, default=0,
                        help="length of the context on each side")
    ARGS: Namespace = PARSER.parse_args()

    main(ARGS.datadir, ARGS.mode, ARGS.train_size, ARGS.validation_size,
         vars(ARGS))
