# pylint: disable=missing-docstring
import argparse
from collections import Counter
import csv
import os


def main(in_filename, out_filename, percentage):
    """TODO: learn_type_to_class docstring"""
    with open(in_filename, newline='') as csvfile,\
         open(out_filename, mode='w') as vocabfile:  # noqa: E127
        reader = csv.reader(csvfile, delimiter=',')
        types = tuple(map(lambda t: t[1], reader))[1:]
        included = 0
        for typ, count in Counter(types).most_common():
            print(typ, file=vocabfile)
            included += count
            if included / len(types) > percentage:
                return


if __name__ == '__main__':
    # Get program arguments
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('in_filename', type=str,
                        help='filename of the training set to count types in')
    parser.add_argument('--out_filename', default=None, type=str,
                        help='filename of the vocabulary file to write to')
    parser.add_argument('--percentage', default=0.9, type=float,
                        help='least percentage of types in vocabulary')
    args = parser.parse_args()  # pylint: disable=invalid-name

    # pylint: disable=invalid-name
    out_name = args.out_filename if args.out_filename is not None\
               else os.path.join(  # noqa: E127
                       os.path.dirname(os.path.abspath(args.in_filename)),
                       "vocab.txt")

    main(args.in_filename, out_name, args.percentage)
