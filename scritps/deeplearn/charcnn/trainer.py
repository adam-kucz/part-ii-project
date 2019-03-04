from datetime import datetime, timedelta
from pathlib import Path

from parse import parse
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard)

from ..abstract.modules import (DataMode, DataReader)
from .input_output import get_out_dir

__all__ = ['Trainer']

ZERODATE = datetime(1970, 1, 1)


class Trainer:
    name: str
    data_reader: DataReader
    _model: Model
    _epoch: int
    out_dir: Path

    def __init__(self, name: str, data_reader: DataReader, model: Model,
                 out_dir: Path, append_format: str = ''):
        self.name = name
        self.data_reader = data_reader
        self._model = model
        self._epoch = 0
        self.out_dir = get_out_dir(out_dir, model.get_config())
        self.fileformat = self.name + '.{epoch:04d}' + append_format

    def train(self, trainpath, valpath, batch_size=100, epochs=100, verbose=1):
        train_dataset = self.data_reader(trainpath, DataMode.TRAIN)
        val_dataset = self.data_reader(valpath, DataMode.TEST)
        result = self.model.fit(
            x=train_dataset, batch_size=batch_size,
            epochs=epochs, verbose=verbose,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(self.fileformat,
                                save_weights_only=True, period=50),
                TensorBoard(log_dir=self.out_dir, write_images=True)],
            validation_data=val_dataset, initial_epoch=self.epoch,
            steps_per_epoch=None, validation_steps=None)
        self._epoch += epochs
        return result

    @property
    def model(self):
        return self._model

    @property
    def epoch(self):
        return self._epoch

    def load_weights(self):
        try:
            rich = map(lambda p: (p, parse(self.fileformat, str(p)),
                                  p.stat().st_mtime),
                       self.out_dir.iterdir())
            path, parsed, time = max(filter(lambda t: t[1], rich),
                                     key=lambda t: t[2])
            self.model.load_weights(path)
            date = ZERODATE + timedelta(seconds=time)
            print("Loaded wieghts from file {}, saved at {} with epoch {}"
                  .format(path, date, parsed['epoch']))
            self._epoch = parsed['epoch']
        except ValueError:
            raise ValueError("Cannot load weights, no saved file found")
