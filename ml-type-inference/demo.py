# pylint: disable=missing-docstring
import argparse

import identifier_type_data as data
from onedcnn import CNN1d


def main(num_epochs, batch_size=100, learn_rate=0.01):
    """TODO: describe main"""
    # Prepare and fetch the data
    data_loader = data.DataLoader()
    data_loader.load_data()

    # Build a CNN with 5 hidden layers
    params = {'identifier_len': 12,
              'num_chars_in_vocab': data_loader.num_chars_in_vocabulary,
              'convolutional': [{'filters': 32, 'kernel_size': 5},
                                {'filters': 32, 'kernel_size': 5},
                                {'filters': 32, 'kernel_size': 3}],
              'dense': [{'units': 64},
                        {'units': 64}],
              'n_classes': data_loader.num_classes}

    with CNN1d(params, "./out") as network:
        # Train the Model.
        network.train(num_epochs,
                      lambda _: data_loader.training_data(batch_size),
                      lambda _: learn_rate)

        # Evaluate the model.
        metrics = network.test(*data_loader.validation_data(batch_size))

        print('\nValidate set accuracy: {accuracy:0.3f}\n'.format(**metrics))


if __name__ == '__main__':
    # Get program arguments
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--train_steps', default=1000, type=int,
                        help='number of training steps')
    args = parser.parse_args()  # pylint: disable=invalid-name

    main(args.train_steps, args.batch_size)
