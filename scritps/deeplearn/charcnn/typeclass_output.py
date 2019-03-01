from typing import Callable, Mapping, Tuple

import tensorflow as tf

from ..abstract.modules import OutputNet, Collections

__all__ = ['typeclass_output', 'TypeclassOutput']


def typeclass_output(outputs: tf.Tensor,
                     one_hot_labels: tf.Tensor,
                     class_num: int,
                     log: Callable[[str], None] = lambda _: None):
    return TypeclassOutput(class_num, log)((outputs,), (one_hot_labels,))


class TypeclassOutput(OutputNet):
    _epoch: tf.Tensor
    _increment_epoch: tf.Tensor
    _learning_rate: tf.Tensor
    _train_op: tf.Operation
    _metric_vals: Mapping[str, tf.Tensor]
    _metric_ops: Mapping[str, tf.Tensor]
    _class_num: int
    # mypy gets confused when this is uncommented
    # _log: Callable[[str], None]

    def __init__(self,
                 class_num: int,
                 log: Callable[[str], None] = lambda _: None):
        self._metric_vals = {}
        self._metric_ops = {}
        self._class_num = class_num
        self._log = log

    def __call__(self,
                 outputs: Tuple[tf.Tensor, ...],
                 label_tensors: Tuple[tf.Tensor, ...],
                 vcs: Collections = None) -> 'TypeclassOutput':
        one_hot_labels = label_tensors[0]

        with tf.name_scope("counters"):
            tf.train.create_global_step()
            self._epoch = tf.get_variable('epoch', (), tf.int32,
                                          initializer=tf.zeros_initializer())
            self._increment_epoch = self._epoch.assign_add(1)

        with tf.name_scope("labels"):
            labels = tf.math.argmax(one_hot_labels, 1)
            self._log("Shape of one_hot_labels: {}, labels: {}"
                      .format(one_hot_labels.shape, labels.shape))

        with tf.name_scope("output"):
            final_layer = tf.keras.layers.Dense(self._class_num,
                                                activation=None)
            tf.get_default_graph().add_to_collections(
                vcs, final_layer.trainable_variables)
            logits = final_layer(outputs[0])
            predictions = tf.argmax(logits, 1)

        with tf.name_scope("training"):
            self._log("Shape of labels: {}, logits: {}"
                      .format(labels.shape, logits.shape))
            loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
            self._learning_rate = tf.placeholder(tf.float32, shape=(),
                                                 name='learning_rate')
            self._train_op = tf.train.AdagradOptimizer(self._learning_rate)\
                                     .minimize(loss,
                                               tf.train.get_global_step())

        with tf.name_scope("metrics"):
            self._add_metric('loss', tf.metrics.mean(loss))
            self._add_metric('accuracy',
                             tf.metrics.accuracy(labels, predictions))
            self._add_metric('real_accuracy',
                             tf.metrics.accuracy(
                                 labels, predictions,
                                 tf.not_equal(one_hot_labels[:, -1], 1)))
            self._add_metric(
                'top5', tf.metrics.mean(tf.nn.in_top_k(logits, labels, 5)))
            self._add_metric(
                'top3', tf.metrics.mean(tf.nn.in_top_k(logits, labels, 3)))

        return self

    def _add_metric(self, name, metric):
        metric_value, metric_op = metric
        self.metric_vals[name] = metric_value
        self.metric_ops[name] = metric_op

    @property
    def epoch(self):
        return self._epoch

    @property
    def increment_epoch(self):
        return self._increment_epoch

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def train_op(self):
        return self._train_op

    @property
    def metric_vals(self):
        return self._metric_vals

    @property
    def metric_ops(self):
        return self._metric_ops

    @property
    def params(self) -> Mapping[str, int]:
        return {'classnum': self._class_num}
