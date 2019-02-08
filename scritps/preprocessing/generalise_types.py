# pylint: disable=missing-docstring
import argparse
import csv
from pathlib import Path
from typing import List, Optional

from type_representation import Type


def generalize_to_vocab(vocab: List[str], typ_str: str) -> str:
    def in_vocab(typ_str: str) -> Optional[str]:
        generalized = Type.from_str(typ_str)
        while generalized:
            if str(generalized) in vocab:
                return str(generalized)
            generalized = generalized.generalize()
        return None
    found = in_vocab(typ_str)
    return found if found else typ_str


def main(vocab_filename: Path, in_filename: Path, out_filename: Path):
    """TODO: learn_type_to_class docstring"""
    with in_filename.open(newline='') as infile,\
         out_filename.open(mode='w') as outfile:  # noqa: E127
        vocab = vocab_filename.read_text().split('\n')
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        count, in_v, new_in_v = 0, 0, 0
        for row in reader:
            count += 1
            general = generalize_to_vocab(vocab, row[1])
            if row[1] in vocab:
                in_v += 1
            if general in vocab:
                new_in_v += 1
            # if general != row[1]:
                # print("Generalized '{}' to '{}'".format(row[1], general))
            writer.writerow([row[0], general])
    print("Went from {}/{count} ({}%) to {}/{count} ({}%)"
          .format(in_v, 100 * in_v / count, new_in_v, 100 * new_in_v / count,
                  count=count))


if __name__ == '__main__':
    # Get program arguments
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('in_filename', type=Path,
                        help='input dataset filename')
    parser.add_argument('--vocab_filename', default=Path('vocab.txt'),
                        type=Path,
                        help='filename of the vocabulary file to write to')
    parser.add_argument('--out_filename', default=None, type=Path,
                        help='output dataset filename')
    args = parser.parse_args()  # pylint: disable=invalid-name

    in_name = args.in_filename  # pylint: disable=invalid-name
    out_name = args.out_filename  # pylint: disable=invalid-name
    if not out_name:
        out_name = in_name.parent.joinpath(in_name.stem + '-general')\
                                 .with_suffix(in_name.suffix)

    main(args.vocab_filename, args.in_filename, out_name)
