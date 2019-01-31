# pylint: disable=missing-docstring
import argparse

import identifier_type_data as data
from onedcnn import CNN1d


IDENTIFIER_LENGTH = 15


def main(num_epochs, batch_size=100, learn_rate=0.01):
    """TODO: describe main"""
    # Prepare and fetch the data
    data_loader = data.DataLoader(IDENTIFIER_LENGTH)
    train_dataset = data_loader.train_dataset
    print("Train data: (type: {})\n{}"
          .format(type(train_dataset), train_dataset))
    return

    # Build a CNN with 5 hidden layers
    params = {'identifier_len': IDENTIFIER_LENGTH,
              'num_chars_in_vocab': data_loader.num_chars_in_vocabulary,
              'convolutional': [{'filters': 32, 'kernel_size': 5},
                                {'filters': 32, 'kernel_size': 5},
                                {'filters': 32, 'kernel_size': 3}],
              'dense': [{'units': 64},
                        {'units': 64}],
              'n_classes': data_loader.num_classes}

    with CNN1d(params, "./out") as network:
        # Train the Model.
        iterator = data_loader.training_data(batch_size)\
                              .make_initializable_iterator()
        network.train(num_epochs, iterator, lambda _: learn_rate)

        # Evaluate the model.
        validation_data = data_loader.validation_data(batch_size)\
                                     .make_one_shot_iterator()\
                                     .get_next()
        metrics = network.test(validation_data)

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
