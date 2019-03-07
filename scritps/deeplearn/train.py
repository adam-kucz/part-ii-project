import argparse
import os
from pathlib import Path
from typing import Callable

import tensorflow as tf

from .modules.model_trainer import ModelTrainer

__all__ = ['interactive']

OUT_PATH: Path = Path("/home/acalc79/synced/part-ii-project/out")

# defaults to 0
# 0 - all logs shown
# 1 - no INFO logs
# 2 - no WARNING logs
# 3 - no ERROR logs
# source: https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints#40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def interactive(trainer_producer: Callable[[int, dict], ModelTrainer],
                data_path: Path, train_name: str = 'train.csv',
                validate_name: str = 'validate.csv',
                final_checkpoint: str = 'final.keras',
                out_path: Path = OUT_PATH):
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('--run_name', default='default', type=str,
                        help='unique name for the run')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--train_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='learning rate [ignored]')
    parser.add_argument('-t', '--test', action='store_true',
                        help='test only (on the validation set)')
    args = parser.parse_args()  # pylint: disable=invalid-name

    train_path: Path = data_path.joinpath(train_name)
    validate_path: Path = data_path.joinpath(validate_name)

    trainer = trainer_producer(args.batch_size,
                               out_dir=out_path, run_name=args.run_name)
    try:
        trainer.load_weights(final_checkpoint)
        print("Successfully restored network with epoch {}"
              .format(trainer.epoch))
    except ValueError:
        print("No checkpoints found, training from scratch")

    if not args.test:
        trainer.train(train_path, validate_path, args.train_epochs)
        trainer.save_weights(final_checkpoint)
    trainer.test(validate_path)
