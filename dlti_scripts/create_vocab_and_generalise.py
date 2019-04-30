import argparse
from pathlib import Path

from funcy import lremove

from preprocessing.sets.type_to_class import create_vocab
from preprocessing.sets.type_generalisation import generalise_file


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_set_path', type=Path,
                        help='training dataset filepath')
    parser.add_argument('-p', '--percentage', type=float, default=0.9,
                        help='least percentage of types in vocabulary')
    parser.add_argument('-v', '--vocab_path', type=Path,
                        default=Path('vocab.csv'),
                        help='filename of the vocabulary file to write to')
    parser.add_argument('-g', '--generalise', nargs='*', type=Path,
                        default=None,
                        help=("datasets to generalise, "
                              "generalises all .csv in the training set "
                              "directory if unspecified"))
    parser.add_argument('-a', '--avoid', nargs='*', type=Path,
                        default=[],
                        help=("datasets not to generalise, "
                              "overrides -g in conflicting cases"))
    parser.add_argument('-s', '--suffix', type=str, default="-general",
                        help="suffix to add to generalised dataset names")
    parser.add_argument('--verbose', action='store_true',
                        help='print summary of actions done')
    args = parser.parse_args()

    vocab_path = args.train_set_path.parent.joinpath(args.vocab_path)
    vocab_size, total = create_vocab(args.train_set_path, vocab_path,
                                     args.percentage)
    if args.verbose:
        print("{} types are needed to cover {}% of {} types in {}"
              .format(vocab_size, 100 * args.percentage,
                      total, args.train_set_path))
    generalise = lremove(
        lambda p: p in args.avoid or Path(p.name) in args.avoid,
        (args.generalise
         or args.train_set_path.resolve().parent.rglob("*.csv")))
    for path in generalise:
        general_name = Path(path.stem + args.suffix).with_suffix(path.suffix)
        in_v, new_in_v, count = generalise_file(
            vocab_path, path, path.parent.joinpath(general_name), verbose=0)
        if args.verbose:
            print("Went from {}/{count} ({}%) to {}/{count} ({}%)"
                  " for dataset {}"
                  .format(in_v, 100 * in_v / count, new_in_v,
                          100 * new_in_v / count, general_name, count=count))


if __name__ == '__main__':
    main()
