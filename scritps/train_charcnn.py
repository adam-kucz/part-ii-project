import argparse
from pathlib import Path

from deeplearn.charcnn.standalone import FullCharCNN


DATA_DIR: Path = Path("/home/acalc79/synced/part-ii-project/"
                      "data/sets/pairs_funs_as_ret")
VOCAB_PATH: Path = DATA_DIR.joinpath("vocab.txt")
TRAIN_PATH: Path = DATA_DIR.joinpath("train.csv")
VALIDATE_PATH: Path = DATA_DIR.joinpath("validate.csv")
OUT_PATH: Path = Path("/home/acalc79/synced/part-ii-project/out")
IDENTIFIER_LEN: int = 15


def main(num_epochs, batch_size, out_path=OUT_PATH):
    # Build a CNN with 6 hidden layers
    params = {'identifier_length': IDENTIFIER_LEN,
              'convolutional': [{'filters': 32, 'kernel_size': 3},
                                {'filters': 32, 'kernel_size': 3},
                                {'filters': 24, 'kernel_size': 3},
                                {'filters': 16, 'kernel_size': 3}],
              'dense': [{'units': 48},
                        {'units': 48}]}

    net = FullCharCNN(VOCAB_PATH, params, out_path)
    try:
        net.load_weights()
        print("Successfully restored network with epoch {}"
              .format(net.epoch))
    except ValueError:
        print("No checkpoints found, training from scratch")

    net.train(TRAIN_PATH, VALIDATE_PATH, batch_size, num_epochs, verbose=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('--run_name', default=None, type=str,
                        help='unique name for the run [ignored]')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--train_epochs', default=20, type=int,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='learning rate [ignored]')
    args = parser.parse_args()  # pylint: disable=invalid-name

    main(args.train_epochs, args.batch_size)
