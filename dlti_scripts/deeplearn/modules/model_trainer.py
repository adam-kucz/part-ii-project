from pathlib import Path
from typing import Optional

import numpy as np
from parse import parse
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.callbacks as cb

from ..abstract import (DataMode, DataReader)

__all__ = ['ModelTrainer']


def _steps_per_epoch(dataset: tf.data.Dataset) -> int:
    # hack to get dataset size requires disabling some pylint rules
    DatasetV1Adapter = type(tf.data.Dataset.range(0))
    # pylint: disable=invalid-name,protected-access
    RepeatDataset = type(tf.data.Dataset.range(0).repeat()._dataset)
    # pylint: enable=invalid-name
    if isinstance(dataset, DatasetV1Adapter):
        dataset = dataset._dataset
    while isinstance(dataset, RepeatDataset):
        dataset = dataset._input_dataset
        if isinstance(dataset, DatasetV1Adapter):
            dataset = dataset._dataset
    # pylint: enable=protected-access

    result = dataset.reduce(0, lambda x, _: x + 1)
    sess = tf.Session()
    sess.run(tf.initializers.tables_initializer())
    result = sess.run(result)
    return result


class RestoreBest(cb.Callback):
    monitor: str
    best: float
    verbose: int
    best_epoch: int

    def __init__(self, monitor='val_loss', verbose=0, mode='auto'):
        super().__init__()
        self.monitor = monitor

        self.verbose = verbose
        self.best_epoch = 0
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            raise ValueError('RestoreBest mode {} is unknown'.format(mode))

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        self.best_epoch = 0
        # pylint: disable=comparison-with-callable
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_train_end(self, logs=None):  # pylint: disable=unused-argument
        if self.best_epoch > 0:
            self.model.set_weights(self.best_weights)
            if self.verbose > 0:
                print('Restoring best weights from epoch {}'
                      .format(self.best_epoch))

    def get_monitor_value(self, logs={}):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            raise ValueError(('Early stopping conditioned on metric `{}` '
                              'which is not available.'
                              'Available metrics are: {}')
                             .format(self.monitor, ','.join(logs)))
        return monitor_value


class ModelTrainer:
    name: str
    data_reader: DataReader
    outpath: Path
    _model: Model
    _core: Model
    _epoch: int
    fileformat: str
    _run_name: str
    _checkpointpath: Path
    _checkpointformat: Path

    def __init__(self, name: str,
                 data_reader: DataReader, model: Model, core: Model,
                 outpath: Path, run_name: str = 'default',
                 append_format: str = '', monitor: str = 'val_loss'):
        self.name = name
        self.data_reader = data_reader
        self._model = model
        self._core = core
        self._epoch = 0
        self.outpath = outpath
        self.fileformat = self.name + '.{epoch:04d}' + append_format
        self.run_name = run_name
        self.monitor = monitor

    def _ensure_initialized(self):
        sess = K.get_session()
        try:
            sess.run(tf.initializers.tables_initializer())
        except tf.errors.FailedPreconditionError:
            pass

    def train(self, trainpath, valpath, epochs=100, patience=64, verbose=1):
        self._ensure_initialized()
        train_dataset = self.data_reader(trainpath, DataMode.TRAIN)
        val_dataset = self.data_reader(valpath, DataMode.VALIDATE)
        log_dir = str(self.outpath.joinpath(self.run_name, "tensorboard"))
        result = self.model.fit(
            x=train_dataset,
            initial_epoch=self.epoch, epochs=self.epoch + epochs,
            callbacks=[
                cb.ModelCheckpoint(self._checkpointformat,
                                   save_weights_only=True, period=50,
                                   verbose=verbose),
                cb.TensorBoard(log_dir=log_dir, write_images=True),
                cb.EarlyStopping(monitor=self.monitor, patience=patience,
                                 restore_best_weights=True,
                                 verbose=verbose),
                RestoreBest(monitor=self.monitor, verbose=verbose)],
            verbose=verbose, shuffle=False, validation_data=val_dataset,
            steps_per_epoch=_steps_per_epoch(train_dataset),
            validation_steps=_steps_per_epoch(val_dataset))
        self._epoch += epochs
        return result

    def test(self, valpath: Path, verbose=1):
        self._ensure_initialized()
        val_dataset = self.data_reader(valpath, DataMode.TEST)
        self.model.evaluate(x=val_dataset, verbose=verbose,
                            steps=_steps_per_epoch(val_dataset))

    @property
    def model(self):
        return self._model

    @property
    def epoch(self):
        return self._epoch

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, value: str):
        self._run_name = value
        self._checkpointpath = self.outpath.joinpath(self.run_name,
                                                     "checkpoints")
        if not self._checkpointpath.exists():
            self._checkpointpath.mkdir(parents=True)
        self._checkpointformat = str(
            self._checkpointpath.joinpath(self.fileformat))

    def load_weights(self, filename: Optional[str] = None):
        found = None
        if filename:
            path = self._checkpointpath.joinpath(filename)
            if path.exists():
                found = str(path)
        else:
            # TODO: remove '.index' hack
            for filepath in self._checkpointpath.iterdir():
                relpath = filepath.relative_to(self._checkpointpath)
                parsed = parse(self.fileformat + '.index', str(relpath))
                if parsed and parsed['epoch'] > self.epoch:
                    found = str(filepath)
                    self._epoch = parsed['epoch']
            found = found[:-len('.index')]

        if not found:
            raise ValueError("Cannot load weights, no saved file found")
        self.model.load_weights(found)
        print("Loaded wieghts from epoch {}, file {}"
              .format(self.epoch, found))

    def save_weights(self, filename):
        self.model.save_weights(str(self._checkpointpath.joinpath(filename)))

    def save_core_weights(self, filename):
        self._core.save_weights(str(self._checkpointpath.joinpath(filename)))
