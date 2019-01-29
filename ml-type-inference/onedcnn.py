"""TODO"""
import os
from typing import Mapping

import numpy as np
import tensorflow as tf


class CNN1d:
    """TODO: CNN1d docstring"""
    out_dir: str
    features: tf.Tensor
    labels: tf.Tensor
    outputs: Mapping[str, tf.Tensor]
    metrics: Mapping[str, tf.Tensor]
    learning_rate: tf.Tensor
    train_op: tf.Operation
    summary: tf.Tensor
    _sess: tf.Session

    def __init__(self, params, out_dir) -> None:
        self.out_dir = out_dir
        tf.train.create_global_step()
        self._build_network(params)
        self._sess = tf.Session()

    def _build_network(self, params):
        # Create input layer.
        # dimensions = batch_size x max_chars
        self.features = tf.placeholder(tf.uint8,
                                       shape=(None, params['identifier_len']),
                                       name='features')
        one_hot_chars = tf.one_hot(self.features,
                                   depth=params['num_chars_in_vocab'])
        print("Shape of features: {}, one_hot: {}"
              .format(self.features.shape, one_hot_chars.shape))

        # Create 1d convolutional layers.
        with tf.name_scope("conv"):
            tensor = one_hot_chars
            for conv in params['convolutional']:
                tensor = tf.layers.conv1d(inputs=tensor,
                                          filters=conv['filters'],
                                          kernel_size=conv['kernel_size'],
                                          padding='valid',
                                          use_bias=False,
                                          activation=tf.nn.relu)
                print("Shape of tensor after convolution: {}"
                      .format(tensor.shape))

        # Create dense layers.
        with tf.name_scope("dense"):
            tensor = tf.layers.flatten(tensor)
            for dense in params['dense']:
                tensor = tf.layers.dense(inputs=tensor,
                                         units=dense['units'],
                                         activation=tf.nn.relu)
                print("Shape of tensor after dense: {}".format(tensor.shape))

        with tf.name_scope("output"):
            # Compute logits (1 per class).
            logits = tf.layers.dense(tensor, params['n_classes'],
                                     activation=None)
            predictions = tf.argmax(logits, 1)
            print("Shape of logits: {}, predictions: {}"
                  .format(logits.shape, predictions.shape))
            self.outputs = {'logits': logits,
                            'predictions': predictions}

        self.labels = tf.placeholder(tf.int32, shape=(None,))

        with tf.name_scope("metrics"):
            print("Shape of labels: {}, logits: {}"
                  .format(self.labels.shape, logits.shape))
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=self.labels,
                logits=logits)
            accuracy = tf.metrics.accuracy(labels=self.labels,
                                           predictions=predictions,
                                           name='acc_op')
            self.metrics = {'accuracy': accuracy,
                            'loss': loss}

        # Write summary.
        with tf.variable_scope("logging"):
            for name, metric in self.metrics.items():
                tf.summary.scalar(name, metric)
            self.summary = tf.summary.merge_all()

        # Create training op.
        self.learning_rate = tf.placeholder(tf.float32, shape=(),
                                            name='learning_rate')
        self.train_op = tf.train.AdagradOptimizer(self.learning_rate)\
                                .minimize(loss, tf.train.get_global_step())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Free resources used by the network"""
        self._sess.close()

    def train_step(self, features, labels, learning_rate, writer=None) -> None:
        """TODO: train_step docstring"""
        self._sess.run(tf.global_variables_initializer())
        _, metrics, summary, step = self._sess.run(
            [self.train_op, self.metrics,
             self.summary, tf.train.get_global_step()],
            feed_dict={self.learning_rate: learning_rate,
                       self.features: features,
                       self.labels: labels})
        if writer:
            writer.add_summary(summary, step)
        return metrics

    def train(self, num_epochs, batcher, learning_rate):
        """TODO: train docstring"""
        epoch_logs = []
        summary_dir = os.path.join(self.out_dir, "summaries", "train")
        with tf.summary.FileWriter(summary_dir, self._sess.graph) as writer:
            for epoch in range(num_epochs):
                epoch_metrics = {'accuracy': 0, 'loss': 0}
                rate = learning_rate(epoch)
                batches = batcher(epoch)
                for batch_data in batches:
                    batch_metrics = self.train_step(*batch_data, rate, writer)
                    for key in batch_metrics:
                        epoch_metrics[key] += batch_metrics[key] / len(batches)
                epoch_logs.append(epoch_metrics)
                writer.flush()
        # TODO: consider changing into a generator
        return epoch_logs

    def test(self, features, labels) -> None:
        """TODO: test docstring"""
        self._sess.run(tf.global_variables_initializer())
        return self._sess.run(self.metrics, feed_dict={self.features: features,
                                                       self.labels: labels})

    def predict(self, features) -> np.ndarray:
        """TODO: predict docstring"""
        self._sess.run(tf.global_variables_initializer())
        outputs = self._sess.run(self.outputs,
                                 feed_dict={self.features: features})
        return {'class_ids': outputs['predictions'][:, tf.newaxis],
                'probabilities': tf.nn.softmax(outputs['logits']),
                'logits': outputs['logits']}

    def save_checkpoint(self, max_num_checkpoints) -> None:
        """Saves model with trained parameters"""
        checkpoint_dir = os.path.abspath(os.path.join(self.out_dir,
                                                      "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=max_num_checkpoints)
        saver.save(self._sess, checkpoint_prefix,
                   global_step=tf.train.get_global_step().eval())

    def restore_checkpoint(self, checkpoint: str) -> None:
        """Reads saved model from file"""
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self._sess, checkpoint)