from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import Model

from ..abstract.modules import (DataMode, DataReader)
from .input_output import IO

__all__ = ['Trainer']


class Trainer:
    data_reader: DataReader
    _model: Model
    # pylint: disable=invalid-name
    io: IO

    def __init__(self, data_reader: DataReader, model: Model, out_dir: Path):
        self.data_reader = data_reader
        self._model = model
        self.io = IO(out_dir, model.params, {}, self)

    def train(self, trainpath, valpath, batch_size=100, epochs=100, verbose=1):
        train_dataset = self.data_reader(trainpath, DataMode.TRAIN)
        val_dataset = self.data_reader(valpath, DataMode.TEST)
        return self.model.fit(dataset=train_dataset, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              callbacks=[], validation_data=val_dataset,
                              initial_epoch=0, steps_per_epoch=None,
                              validation_setps=None)

    @property
    def model(self):
        return self._model

    def close(self):
        self.io.close()

    def save_checkpoint(self, max_num_checkpoints: int = 10) -> None:
        """Saves model with trained parameters"""
        self.io.save_checkpoint(max_num_checkpoints)

    def restore_checkpoint(self, checkpoint: Optional[str] = None) -> None:
        """Reads saved model from file"""
        self.io.restore_checkpoint(checkpoint)
