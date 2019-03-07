from pathlib import Path

from deeplearn.modules.charcnn import CharCNN
from deeplearn.standalone.networks import FullContextNet
import deeplearn.train

DATA_DIR: Path = Path("/home/acalc79/synced/part-ii-project/"
                      "data/sets/ctx_trivial")
VOCAB_PATH: Path = DATA_DIR.joinpath("vocab.txt")
IDENTIFIER_LEN: int = 15


def main(data_dir: Path):
    # Build a ContextNet
    charcnn = {'identifier_length': IDENTIFIER_LEN,
               'convolutional': [{'filters': 32, 'kernel_size': 3},
                                 {'filters': 32, 'kernel_size': 3},
                                 {'filters': 24, 'kernel_size': 3},
                                 {'filters': 16, 'kernel_size': 3}],
               'dense': [{'units': 48},
                         {'units': 48}]}
    contextnet = {'ctx_len': 5,
                  'token_net': CharCNN(charcnn),
                  'aggregate': [{'units': 32}]}

    def trainer_producer(batch_size, **kwargs):
        return FullContextNet(VOCAB_PATH, batch_size, contextnet, **kwargs)

    deeplearn.train.interactive(trainer_producer, data_dir)


if __name__ == '__main__':
    main(DATA_DIR)
