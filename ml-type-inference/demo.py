# pylint: disable=missing-docstring
import argparse
import os

import identifier_type_data as data
from onedcnn import CNN1d

TRAIN_PATH = os.path.join(data.DIR, "train.csv")
VALIDATE_PATH = os.path.join(data.DIR, "validate.csv")
IDENTIFIER_LENGTH = 15


def main(num_epochs, batch_size, learn_rate):
    """TODO: describe main"""
    # Build a CNN with 5 hidden layers
    params = {'train_filepath': TRAIN_PATH,
              'validate_filepath': VALIDATE_PATH,
              'batch_size': batch_size,
              'net': {
                  'identifier_len': IDENTIFIER_LENGTH,
                  'convolutional': [{'filters': 32, 'kernel_size': 3},
                                    {'filters': 32, 'kernel_size': 3},
                                    {'filters': 24, 'kernel_size': 3},
                                    {'filters': 16, 'kernel_size': 3}],
                  'dense': [{'units': 24},
                            {'units': 24},
                            {'units': 24},
                            {'units': 24}]}}

    with CNN1d(params, "/home/acalc79/synced/part-ii-project/out") as network:
        # Train the Model.
        print("Running {} epochs".format(num_epochs))

        def dynamic_learn_rate(epoch):
            return learn_rate * 5 / (50 + epoch)

        for i, epoch_metrics in enumerate(network.train(num_epochs,
                                                        dynamic_learn_rate)):
            print("Metrics after {}th epoch: {}".format(i, epoch_metrics))

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
