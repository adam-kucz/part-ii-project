from collections import Counter
import csv
import os


def create_vocab_file(in_filename, out_filename=None, percentage=0.8):
    """TODO: learn_type_to_class docstring"""
    if out_filename is None:
        dirname = os.path.dirname(os.path.abspath(in_filename))
        out_filename = os.path.join(dirname, "vocab.txt")
    with open(in_filename, newline='') as csvfile,\
         open(out_filename, mode='w') as vocabfile:  # noqa: E127
        reader = csv.reader(csvfile, delimiter=',')
        types = tuple(map(lambda t: t[1], reader))
        included = 0
        for typ, count in Counter(types).items():
            print(typ, file=vocabfile)
            included += count
            if included / len(types) > percentage:
                return
