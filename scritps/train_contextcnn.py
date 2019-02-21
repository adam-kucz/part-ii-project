import argparse
from pathlib import Path

from deeplearn.contextcnn.standalone import FullContextCNN


DATA_DIR: Path = Path("/home/acalc79/synced/part-ii-project" +  # noqa: W504
                      "/data/sets/pairs_funs_as_ret")
VOCAB_PATH: Path = DATA_DIR.joinpath("vocab.txt")
TRAIN_PATH: Path = DATA_DIR.joinpath("train.csv")
VALIDATE_PATH: Path = DATA_DIR.joinpath("validate.csv")
OUT_PATH: Path = Path("/home/acalc79/synced/part-ii-project/out")
IDENTIFIER_LENGTH: int = 12
CONTEXT_LENGTH: int = 8
CONTEXT_SIZE = 5


def main(num_epochs, batch_size, learn_rate, run_name, out_path=OUT_PATH):
    central_charcnn = {'convolutional': [{'filters': 32, 'kernel_size': 3},
                                         {'filters': 24, 'kernel_size': 3},
                                         {'filters': 20, 'kernel_size': 3},
                                         {'filters': 16, 'kernel_size': 3}],
                       'dense': [{'units': 24},
                                 {'units': 16}]}
    context_charcnn = {'convolutional': [{'filters': 20, 'kernel_size': 3},
                                         {'filters': 15, 'kernel_size': 3},
                                         {'filters': 10, 'kernel_size': 3}],
                       'dense': [{'units': 16},
                                 {'units': 8}]}
    params = {'center': central_charcnn,
              'context': context_charcnn,
              'aggregate': [{'units': 32},
                            {'units': 24}]}

    with FullContextCNN(VOCAB_PATH, IDENTIFIER_LENGTH, CONTEXT_LENGTH,
                        batch_size, params, out_path, print) as network:
        try:
            network.restore_checkpoint()
            print("Successfully restored network with epoch {}"
                  .format(network.epoch))
        except ValueError:
            print("No checkpoints found, training from scratch")

        # Train the Model.
        def dynamic_lr(epoch):
            return learn_rate * 5 / (50 + epoch)

        for metrics in network.train_epochs(TRAIN_PATH,
                                            VALIDATE_PATH,
                                            num_epochs,
                                            dynamic_lr,
                                            run_name):
            if network.epoch % 10 == 0:
                print("Metrics after {}th epoch: {}"
                      .format(network.epoch, metrics))

        network.save_checkpoint()

        # Evaluate the model.
        metrics = network.test(VALIDATE_PATH)

        print("\nFinal validation metrics:\n")
        for name, val in metrics.items():
            print('{}: {:0.3f}'.format(name, val))


if __name__ == '__main__':
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
