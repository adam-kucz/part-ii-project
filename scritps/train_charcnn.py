from pathlib import Path

from deeplearn.standalone.networks import FullCharCNN
import deeplearn.train

DATA_DIR: Path = Path("/home/acalc79/synced/part-ii-project/"
                      "data/sets/pairs_funs_as_ret")
VOCAB_PATH: Path = DATA_DIR.joinpath("vocab.txt")
IDENTIFIER_LEN: int = 15


def main(data_dir: Path):
    params = {'identifier_length': IDENTIFIER_LEN,
              'convolutional': [{'filters': 32, 'kernel_size': 3},
                                {'filters': 32, 'kernel_size': 3},
                                {'filters': 24, 'kernel_size': 3},
                                {'filters': 16, 'kernel_size': 3}],
              'dense': [{'units': 48},
                        {'units': 48}]}

    def trainer_producer(batch_size, **kwargs):
        return FullCharCNN(VOCAB_PATH, batch_size, params, **kwargs)

    deeplearn.train.interactive(trainer_producer, data_dir)


if __name__ == '__main__':
    main(DATA_DIR)
