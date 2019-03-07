from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from parse import parse
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard)

from ..abstract import (DataMode, DataReader)
from .input_output import get_out_dir

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


class ModelTrainer:
    name: str
    data_reader: DataReader
    outdir: Path
    _model: Model
    _epoch: int
    fileformat: str
    _run_name: str
    _checkpointpath: Path
    _checkpointformat: Path

    def __init__(self, name: str, data_reader: DataReader, model: Model,
                 outdir: Path, run_name: str = 'default',
                 append_format: str = ''):
        self.name = name
        self.data_reader = data_reader
        self._model = model
        self._epoch = 0
        self.outdir = get_out_dir(outdir, model.get_config())
        self.fileformat = self.name + '.{epoch:04d}' + append_format
        self.run_name = run_name

    def _ensure_initialized(self):
        sess = K.get_session()
        try:
            sess.run(tf.initializers.tables_initializer())
        except tf.errors.FailedPreconditionError:
            pass

    def train(self, trainpath, valpath, epochs=100, verbose=1):
        self._ensure_initialized()
        train_dataset = self.data_reader(trainpath, DataMode.TRAIN)
        val_dataset = self.data_reader(valpath, DataMode.VALIDATE)
        result = self.model.fit(
            x=train_dataset,
            initial_epoch=self.epoch, epochs=self.epoch + epochs,
            callbacks=[
                EarlyStopping(patience=32, restore_best_weights=True,
                              verbose=verbose),
                ModelCheckpoint(self._checkpointformat,
                                save_weights_only=True, period=50,
                                verbose=verbose),
                TensorBoard(log_dir=self.outdir, write_images=True)],
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
        self._checkpointpath = self.outdir.joinpath(self.run_name,
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
