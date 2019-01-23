"""TODO"""
from typing import Mapping, Union

from dpu_utils.mlutils import CharTensorizer
import numpy as np
import tensorflow as tf

from type_conversion import TypeConverter


class CNN1d:
    """TODO"""
    __class_num: int
    __max_input_len: int
    __type_converter: TypeConverter
    __char_tensorizer: CharTensorizer
    __hyperparameters: Mapping[str, Union[int, float]]
    __text_characters: tf.Tensor

    def __init__(self, max_input_len=40, class_num=20) -> None:
        self.__hyperparameters = {}
        self.__class_num = class_num
        self.__max_input_len = max_input_len

    def inference(self) -> None:
        """Build the 1dCNN model

        """
        self.__text_characters\
            = tf.placeholder(dtype=tf.int32,
                             shape=[None, self.__max_input_len],
                             name='text_chars')
        dep: int = self.__char_tensorizer.num_chars_in_vocabulary
        characters_one_hot\
            = tf.one_hot(self.__text_characters, depth=dep)
        characters_one_hot = tf.expand_dims(characters_one_hot, dim=0)

        hyp: Mapping[str, str] = self.__hyperparameters
        conv_l1_layer = tf.layers.Conv1D(filters=hyp['l1_filters'],
                                         kernel_size=hyp['l1_kernel_size'],
                                         activation=tf.nn.relu)
        conv_l1 = conv_l1_layer(characters_one_hot)

        conv_l2_layer = tf.layers.Conv1D(filters=hyp['l2_filters'],
                                         kernel_size=hyp['l2_kernel_size'],
                                         activation=tf.nn.relu)
        self.conv_l2 = conv_l2_layer(conv_l1)

        type_targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, self.__class_num])
        self.types_one_hot = tf.one_hot(type_targets,
                                        depth=self.__class_num + 1)

    def save(self) -> None:
        pass

    def read(self, filename: str) -> None:
        pass

    def init_session(self) -> None:
        pass

    def run(self, identifier: str) -> None:
        """TODO"""
        tensorized: np.ndarray\
            = self.__char_tensorizer.tensorize_str(identifier)
        var_init = tf.global_variables_initializer()
        table_init = tf.tables_initializer()
        with tf.Session() as sess:
            sess.run((var_init, table_init))
            result: np.ndarray\
                = sess.run(self.conv_l2,
                           feed_dict={'text_chars': tensorized})
            print(result)

    def write_graph(self) -> None:
        """TODO"""
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        writer.flush()
