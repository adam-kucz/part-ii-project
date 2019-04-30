import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

from funcy import re_test
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
        final_checkpoint: str = 'weights_{}.keras'):
    start_time: datetime = datetime.now()
    parser = argparse.ArgumentParser(
        description="Run '{}'".format(program_name),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('params', type=Path,
                        help='path to json file with network parameters')
    parser.add_argument('data_path', type=Path,
                        help='path to directory with data files')
    parser.add_argument('out_path', type=Path,
                        help='path to save output in')
    parser.add_argument('-r', '--run_name', default='default', type=str,
                        help='unique name for the run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='set verbosity level')
    args = parser.parse_args()

    test_paths: Iterable[Path] = [p for p in args.data_path.glob("*.csv")
                                  if re_test(r'test\d+$', p.stem)]

    params = json.loads(args.params.read_text())
    trainer = trainer_producer(params, args.data_path, args.batch_size,
                               out_dir=args.out_path, run_name=args.run_name)
    for pattern in (final_checkpoint, None):
        try:
            trainer.load_weights(pattern)
            print("Successfully restored network with epoch {}"
                  .format(trainer.epoch))
            break
        except ValueError as err:
            if err.args[1] != 'not_found':
                raise
    else:
        print("No checkpoints found")
        return

    for test_path in test_paths:
        trainer.test_detail(test_path, test_path.stem + '_epoch{}.csv')

    print("Finished successfully in {}".format(datetime.now() - start_time))
