from pathlib import Path
from typing import Iterable, Optional

from funcy import cut_suffix
import numpy as np
from parse import parse
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.callbacks as cb

from ..abstract import DataMode, DataReader
from ..data_ops.data_interface import CompleteRecordReader
from ..util import csv_write


class RestoreBest(cb.Callback):
    monitor: str
    best: float
    verbose: int
    best_epoch: int

    def __init__(self, trainer, set_epoch=lambda _: None,
                 monitor='val_loss', verbose=0, mode='auto'):
        super().__init__()
        # TODO: remove (hack to prevent loss of partially-trained models)
        self.trainer = trainer
        # end hack
        self.set_epoch = set_epoch
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
            # TODO: remove hack
            if epoch - self.last_saved > 20:
                self.trainer.save_weights('weights_{epoch}-optimizer')
                self.last_saved = epoch
            # end hack

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        self.best_epoch = 0
        # pylint: disable=comparison-with-callable
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        # TODO: remove hack
        self.last_saved = self.trainer.epoch
        # end hack

    def on_train_end(self, logs=None):  # pylint: disable=unused-argument
        if self.best_epoch > 0:
            self.model.set_weights(self.best_weights)
            self.set_epoch(self.best_epoch)
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

    def train(self, trainpath, valpath, epochs=100,
              learning_rate=0.01, patience=64, verbose=1):
        self._ensure_initialized()
        train_dataset = self.data_reader(trainpath, DataMode.TRAIN)
        val_dataset = self.data_reader(valpath, DataMode.VALIDATE)
        log_dir = str(self.outpath.joinpath(self.run_name, "tensorboard"))

        def set_epoch(epoch):
            self._epoch = epoch
        return self.model.fit(
            x=train_dataset.data,
            initial_epoch=self.epoch, epochs=self.epoch + epochs,
            callbacks=[
                cb.ModelCheckpoint(self._checkpointformat,
                                   save_weights_only=True, period=50,
                                   verbose=verbose),
                cb.TensorBoard(log_dir=log_dir),
                cb.LearningRateScheduler(lambda _: learning_rate),
                cb.EarlyStopping(monitor=self.monitor, patience=patience,
                                 restore_best_weights=True,
                                 verbose=verbose),
                RestoreBest(trainer=self,
                            set_epoch=set_epoch, monitor=self.monitor,
                            verbose=verbose)],
            verbose=verbose, shuffle=False, validation_data=val_dataset.data,
            steps_per_epoch=train_dataset.steps_per_epoch,
            validation_steps=val_dataset.steps_per_epoch)

    def test(self, valpath: Path, verbose=1):
        self._ensure_initialized()
        val_dataset, steps = self.data_reader(valpath, DataMode.TEST)
        return self.model.evaluate(x=val_dataset, verbose=verbose,
                                   steps=steps)

    def full_predictions(self, valpath, verbose) -> Iterable[Iterable[float]]:
        original_reader = self.data_reader
        self.data_reader = CompleteRecordReader(original_reader)
        predictions = self.predict(valpath, verbose)
        self.data_reader = original_reader
        return predictions
    
        # dataset, _ = original_reader(
        #     valpath, DataMode.INPUTS | DataMode.LABELS | DataMode.ONEPASS)

        # def add_to_tuple(tup, record):
        #     (xs, ys), (x, y) = tup, record
        #     y = tf.expand_dims(y, 0)
        #     return (tf.concat([xs, x], 0), tf.concat([ys, y], 0))
        # empty_x = tf.constant([], dtype=tf.string)
        # empty_y = tf.constant([], dtype=tf.int64)
        # print("Reducing")
        # xs, ys = K.get_session().run(
        #     dataset.reduce((empty_x, empty_y), add_to_tuple))
        # print("Reduced")
        # self.data_reader = original_reader
        # return zip(xs, ys, predictions)

    def test_detail(self, valpath: Path, out_fileformat: Path, verbose=1):
        self._ensure_initialized()
        val_dataset, steps = self.data_reader(valpath, DataMode.TEST)
        print(f"Dataset: {val_dataset}, steps: {steps}")
        metrics = self.model.evaluate(x=val_dataset, verbose=verbose,
                                      steps=steps)
        out_filename = self.outpath.joinpath(self.run_name,
                                             out_fileformat.format(self.epoch))
        csv_write(out_filename, self.full_predictions(valpath, verbose))
        return metrics

    def predict(self, data_path: Path, verbose=1):
        self._ensure_initialized()
        dataset, steps = self.data_reader(data_path, DataMode.PREDICT)
        return self.model.predict(x=dataset, steps=steps, verbose=verbose)

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

    def load_weights(self, filename_format: Optional[str] = None):
        found = None
        # TODO: deal better with '.index' hacks
        file_format = (filename_format or self.fileformat) + '.index'
        for filepath in self._checkpointpath.iterdir():
            relpath = filepath.relative_to(self._checkpointpath)
            parsed = parse(file_format, str(relpath))
            if not parsed:
                continue
            epoch = int(parsed['epoch'] if 'epoch' in parsed else parsed[0])
            if epoch >= self.epoch:
                found = str(filepath)
                self._epoch = epoch + 1

        if not found:
            raise ValueError("Cannot load weights, no saved file found",
                             'not_found')

        found = cut_suffix(found, '.index')
        self.model.load_weights(found)
        # print("Loaded wieghts from epoch {}, file {}"
        #       .format(self.epoch, found))

    def save_weights(self, filename_format):
        filename = filename_format.format(epoch=self.epoch)
        path = self._checkpointpath.joinpath(filename)
        self.model.save_weights(str(path))

    def save_core_weights(self, filename_format):
        filename = filename_format.format(epoch=self.epoch)
        self._core.save_weights(str(self._checkpointpath.joinpath(filename)))
