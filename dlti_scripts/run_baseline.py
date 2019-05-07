import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Iterable

from funcy import re_test
import tensorflow as tf

from deeplearn.standalone.baseline import Baseline


def trainer_producer(data_path, **kwargs):
    return Baseline(data_path.joinpath("vocab.csv"), **kwargs)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def main(train_name: str = 'train.csv',
         validate_name: str = 'validate.csv',
         test_name: str = 'test.csv'):
    start_time: datetime = datetime.now()
    parser = argparse.ArgumentParser(        
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_path', type=Path,
                        help='path to directory with data files')
    parser.add_argument('out_path', type=Path,
                        help='path to save output in')
    parser.add_argument('-r', '--run_name', default='default', type=str,
                        help='unique name for the run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='batch size')
    args = parser.parse_args()

    train_path: Path = args.data_path.joinpath(train_name)
    test_path: Path = args.data_path.joinpath(test_name)
    trainer = trainer_producer(
        args.data_path, train_path=train_path, batch_size=args.batch_size,
        out_dir=args.out_path, run_name=args.run_name)

    test_paths: Iterable[Path] = [p for p in args.data_path.glob("*.csv")
                                  if re_test(r'test\d+$', p.stem)]
    for test_path in test_paths:
        trainer.test_detail(test_path, test_path.stem + '_epoch{}.csv')

    print("Finished successfully in {}".format(datetime.now() - start_time))


if __name__ == '__main__':
    main()
