import argparse
from pathlib import Path

from preprocessing.sets.type_to_class import create_vocab
from preprocessing.sets.type_generalisation import generalise_file


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()  # pylint: disable=invalid-name
    PARSER.add_argument('train_set_path', type=Path,
                        help='training dataset filepath')
    PARSER.add_argument('-p', '--percentage', type=float, default=0.9,
                        help='least percentage of types in vocabulary')
    PARSER.add_argument('-v', '--vocab_path', type=Path,
                        default=Path('vocab.txt'),
                        help='filename of the vocabulary file to write to')
    PARSER.add_argument('-g', '--generalise', nargs='*', type=Path,
                        help="datasets to generalise, "
                        "generalises all other .csv in the training set "
                        "directory if unspecified")
    PARSER.add_argument('-s', '--suffix', type=str, default="-general",
                        help="suffix to add to generalised dataset names")
    PARSER.add_argument('--verbose', action='store_true',
                        help='print summary of actions done')
    ARGS = PARSER.parse_args()  # pylint: disable=invalid-name

    VOCAB_SIZE, TOTAL = create_vocab(ARGS.train_set_path,
                                     ARGS.vocab_path, ARGS.percentage)
    if ARGS.verbose:
        print("{} types are needed to cover {}% of {} types in {}"
              .format(VOCAB_SIZE, 100 * ARGS.percentage,
                      TOTAL, ARGS.train_set_path))
    GENERALISE = (ARGS.generalise
                  or ARGS.train_set_path.resolve().parent.rglob("*.csv"))
    for path in GENERALISE:
        general_name = Path(path.stem + ARGS.suffix).with_suffix(path.suffix)
        in_v, new_in_v, count = generalise_file(
            ARGS.vocab_path, path, path.parent.joinpath(general_name))
        if ARGS.verbose:
            print("Went from {}/{count} ({}%) to {}/{count} ({}%)"
                  " for dataset {}"
                  .format(in_v, 100 * in_v / count, new_in_v,
                          100 * new_in_v / count, general_name, count=count))
