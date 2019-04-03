from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.models import Model

__all__ = ['OccurenceNet']


class OccurenceNet(Model):
    def __init__(self, params: Mapping[str, Any]):
        
