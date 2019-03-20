import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict

import tensorflow as tf

from .modules.model_trainer import ModelTrainer

__all__ = ['interactive']


# defaults to 0
# 0 - all logs shown
# 1 - no INFO logs
# 2 - no WARNING logs
# 3 - no ERROR logs
# source: https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints#40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def interactive(
        program_name: str,
        trainer_producer: Callable[[int, Dict[str, Any], dict], ModelTrainer],
        train_name: str = 'train.csv',
        validate_name: str = 'validate.csv',
        final_checkpoint: str = 'weights_{}.keras',
        core_checkpoint: str = 'core_weights_{}.keras'):
    parser = argparse.ArgumentParser(
        description="Run '{}'".format(program_name),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('params', type=Path,
                        help='path to json file with network parameters')
    parser.add_argument('data_path', type=Path,
                        help='path to directory with data files')
    parser.add_argument('out_path', type=Path,
                        help='path to save output in')
    parser.add_argument('--run_name', default='default', type=str,
                        help='unique name for the run')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--optimizer', default='adagrad',
                        help='optimizer to use in training')
    # parser.add_argument('--learning_rate', default=0.1, type=float,
    #                     help='learning rate [ignored]')
    parser.add_argument('-t', '--test', action='store_true',
                        help='test only (on the validation set)')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='set verbosity level')
    args = parser.parse_args()

    train_path: Path = args.data_path.joinpath(train_name)
    validate_path: Path = args.data_path.joinpath(validate_name)

    params = json.loads(args.params.read_text())
    try:
        optimizer = tf.keras.optimizers.get(args.optimizer)
    except ValueError:
        optimizer = tf.keras.optimizers.Adam()
    trainer = trainer_producer(params, args.data_path, args.batch_size,
                               out_dir=args.out_path, run_name=args.run_name,
                               optimizer=optimizer)
    try:
        trainer.load_weights(final_checkpoint)
        print("Successfully restored network with epoch {}"
              .format(trainer.epoch))
    except ValueError:
        print("No checkpoints found, training from scratch")

    if not args.test:
        trainer.train(train_path, validate_path,
                      epochs=args.epochs, batch_size=args.batch_size,
                      verbose=args.verbose)
        trainer.save_weights(final_checkpoint)
        trainer.save_core_weights(core_checkpoint)
    trainer.test(validate_path)
