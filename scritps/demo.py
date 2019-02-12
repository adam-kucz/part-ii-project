# pylint: disable=missing-docstring
import argparse
from pathlib import Path

import identifier_type_data as data
from onedcnn import CNN1d

TRAIN_PATH: Path = data.DIR.joinpath("train.csv")
VALIDATE_PATH: Path = data.DIR.joinpath("validate.csv")
OUT_PATH: Path = Path("/home/acalc79/synced/part-ii-project/out")
IDENTIFIER_LENGTH: int = 15


def main(num_epochs, batch_size, learn_rate, run_name, out_path=OUT_PATH):
    """TODO: describe main"""
    # Build a CNN with 5 hidden layers
    params = {'train_filepath': str(TRAIN_PATH),
              'validate_filepath': str(VALIDATE_PATH),
              'batch_size': batch_size,
              'net': {
                  'identifier_len': IDENTIFIER_LENGTH,
                  'convolutional': [{'filters': 32, 'kernel_size': 3},
                                    {'filters': 32, 'kernel_size': 3},
                                    {'filters': 24, 'kernel_size': 3},
                                    {'filters': 16, 'kernel_size': 3}],
                  'dense': [{'units': 48},
                            {'units': 48}]}}

    with CNN1d(params, out_path) as network:
        # Train the Model.
        def dynamic_learn_rate(epoch):
            return learn_rate * 5 / (50 + epoch)

        for i, epoch_metrics in enumerate(network.run(num_epochs,
                                                      dynamic_learn_rate,
                                                      run_name)):
            print("Metrics after {}th epoch: {}".format(i + 1, epoch_metrics))

        # Evaluate the model.
        metrics = network.test()

        print(('\nFinal validate set metrics:\n\n' +
               'loss: {loss:0.3f}\n' +
               'accuracy: {accuracy:0.3f}\n' +
               'real accuracy: {real_accuracy:0.3f}\n' +
               'top3: {top3:0.3f}\n' +
               'top5: {top5:0.3f}\n')
              .format(**metrics))


if __name__ == '__main__':
    # Get program arguments
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('--run_name', default=None, type=str,
                        help='unique name for the run')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--train_epochs', default=20, type=int,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='learning rate for the AdagradOptimizer')
    args = parser.parse_args()  # pylint: disable=invalid-name

    main(args.train_epochs, args.batch_size, args.learning_rate, args.run_name)
