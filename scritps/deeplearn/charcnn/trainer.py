from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import tensorflow as tf

from ..abstract.modules import (DataInterface, DataMode, DataReader,
                                FullNet, OutputNet)
from .input_output import IO
from ..util import merge_parametrized

__all__ = ['Trainer']


class Trainer(tf.Session):
    iter_handle: tf.Tensor
    net: Any

    def __init__(self,
                 data_reader: DataReader,
                 data_interface: DataInterface,
                 network: FullNet,
                 out_dir: Path,
                 graph=None):
        super().__init__(graph=graph)
        self.get_iterator = data_reader
        self.iter_handle: tf.Tensor = tf.placeholder(tf.string, shape=())
        data = data_interface(self.iter_handle)
        print("Data: {}".format(data))
        self.net: OutputNet = network(data)
        params = merge_parametrized(('net', network), ('data', data_interface))
        self.io = IO(out_dir, params, self.net.metric_vals, self)
        self.run(tf.initializers.global_variables())
        self.run(tf.initializers.tables_initializer())

    def close(self):
        self.io.close()
        super().close()

    def train_step(self, handle, learning_rate, step_summary=False) -> None:
        _, metrics = self.run(
            (self.net.train_op, self.net.metric_ops),
            feed_dict={self.iter_handle: handle,
                       self.net.learning_rate: learning_rate})
        if step_summary:
            self.io.write_train_summary(self.step)
        return metrics

    def run_epoch(self,
                  runner: Callable[[], Any],
                  training: bool = True) -> Mapping[str, float]:
        self.run(tf.local_variables_initializer())
        while True:
            try:
                runner()
            except tf.errors.OutOfRangeError:
                if training:
                    self.run(self.net.increment_epoch)
                break
        return self.run(self.net.metric_vals)

    def train_epochs(self,
                     train_path: Path,
                     val_path: Path,
                     num_epochs: int,
                     learning_rate: Callable[[int], float],
                     run_name=None):
        self.io.run_name = run_name
        train_iterator = self.get_iterator(train_path, DataMode.TRAIN)
        val_iterator = self.get_iterator(val_path, DataMode.TEST)
        handle = self.run(train_iterator.string_handle())
        for _ in range(num_epochs):
            self.run(train_iterator.initializer)
            # pylint: disable=cell-var-from-loop
            current_lr = learning_rate(self.run(self.net.epoch))
            metrics = self.run_epoch(
                lambda: self.train_step(handle, current_lr))
            # pylint: disable=no-value-for-parameter
            self.io.write_train_summary(self.epoch)
            self.run(val_iterator.initializer)
            self._test_from_iter(val_iterator)
            yield metrics

    def _test_from_iter(self, test_iterator: tf.data.Iterator)\
            -> Mapping[str, float]:
        handle = self.run(test_iterator.string_handle())
        metrics = self.run_epoch(
            lambda: self.run(
                self.net.metric_ops,
                feed_dict={self.iter_handle: handle}),
            training=False)
        self.io.write_test_summary(self.epoch)
        return metrics

    def test(self, test_path: Path) -> Mapping[str, float]:
        mode = DataMode.TEST | DataMode.ONEPASS
        return self._test_from_iter(self.get_iterator(test_path, mode))

    def save_checkpoint(self, max_num_checkpoints: int = 10) -> None:
        """Saves model with trained parameters"""
        self.io.save_checkpoint(max_num_checkpoints)

    def restore_checkpoint(self, checkpoint: Optional[str] = None) -> None:
        """Reads saved model from file"""
        self.io.restore_checkpoint(checkpoint)

    @property
    def step(self):
        return self.run(tf.train.get_global_step())

    @property
    def epoch(self):
        return self.run(self.net.epoch)
