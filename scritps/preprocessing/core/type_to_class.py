import argparse
from collections import Counter
import csv
from pathlib import Path


def main(in_filename: Path, out_filename: Path, percentage: float):
    with in_filename.open(newline='') as csvfile,\
            out_filename.open(mode='w') as vocabfile:
        reader = csv.reader(csvfile, delimiter=',')
        types = tuple(map(lambda t: t[1], reader))[1:]
        included = 0
        print("Total number of types: {}".format(len(Counter(types))))
        for typ, count in Counter(types).most_common():
            print(typ, file=vocabfile)
            included += count
            if included / len(types) > percentage:
                return


if __name__ == '__main__':
    # Get program arguments
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('in_filename', type=Path,
                        help='filename of the training set to count types in')
    parser.add_argument('--out_filename', default=None, type=Path,
                        help='filename of the vocabulary file to write to')
    parser.add_argument('--percentage', default=0.9, type=float,
                        help='least percentage of types in vocabulary')
    args = parser.parse_args()  # pylint: disable=invalid-name

    # pylint: disable=invalid-name
    out_name = (args.out_filename
                or args.in_filename.parent.joinpath("vocab.txt"))

    main(args.in_filename, out_name, args.percentage)
