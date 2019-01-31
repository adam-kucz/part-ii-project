# pylint: disable=missing-docstring
import argparse
import os

import identifier_type_data as data
from onedcnn import CNN1d

TRAIN_PATH = os.path.join(data.DIR, "train.csv")
VALIDATE_PATH = os.path.join(data.DIR, "validate.csv")
IDENTIFIER_LENGTH = 15


def main(num_epochs, batch_size=100, learn_rate=0.01):
    """TODO: describe main"""
    # Build a CNN with 5 hidden layers
    params = {'train_filepath': TRAIN_PATH,
              'validate_filepath': VALIDATE_PATH,
              'identifier_len': IDENTIFIER_LENGTH,
              'batch_size': batch_size,
              'convolutional': [{'filters': 32, 'kernel_size': 5},
                                {'filters': 32, 'kernel_size': 5},
                                {'filters': 32, 'kernel_size': 3}],
              'dense': [{'units': 64},
                        {'units': 64}]}

    with CNN1d(params, "./out") as network:
        # Train the Model.
        network.train(num_epochs, lambda _: learn_rate)

        # Evaluate the model.
        metrics = network.test()

        print(('\nValidate set accuracy: {accuracy:0.3f}, ' +
               'real accuracy: {real_accuracy:0.3f}, loss: {loss:0.3f}\n')
              .format(**metrics))


if __name__ == '__main__':
    # Get program arguments
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--train_epochs', default=20, type=int,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='learning rate for the AdagradOptimizer')
    args = parser.parse_args()  # pylint: disable=invalid-name

    main(args.train_epochs, args.batch_size, args.learning_rate)
