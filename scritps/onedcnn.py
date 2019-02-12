"""TODO"""
from datetime import datetime
from functools import reduce
from hashlib import md5
import json
import operator as op
from pathlib import Path
from typing import Callable, Mapping
# pylint: disable=unused-import
import sys  # noqa: F401

import tensorflow as tf

from identifier_type_data import DataLoader


class CNN1d:
    """TODO: CNN1d docstring"""
    out_dir: Path
    train_iter: tf.data.Iterator
    val_iter: tf.data.Iterator
    iter_handle: tf.Tensor
    outputs: Mapping[str, tf.Tensor]
    metric_vals: Mapping[str, tf.Tensor]
    metric_ops: Mapping[str, tf.Tensor]
    learning_rate: tf.Tensor
    train_op: tf.Operation
    summary: tf.Tensor
    _sess: tf.Session

    def __init__(self, params: dict, out_dir: Path) -> None:
        self._loader = DataLoader(params['net']['identifier_len'])
        self._data_pipeline(params)
        self._set_out_dir(out_dir, self._loader.vocab, params['net'])
        tf.train.create_global_step()
        self._build_network(params['net'])
        self._sess = tf.Session()
        self._sess.run(tf.initializers.global_variables())
        self._sess.run(tf.initializers.tables_initializer())

    def _set_out_dir(self, out_dir: Path, vocab, net_params):
        encoded = json.dumps((vocab, net_params),
                             sort_keys=True).encode('utf-8')
        identifier = md5(encoded).hexdigest()  # nosec: B303
        subdirs = tuple(out_dir.glob("net{}-*".format(identifier)))
        if subdirs:
            self.out_dir = out_dir.joinpath(out_dir, subdirs[0])
        else:
            time = datetime.now().strftime('%Y-%m-%d-%H-%M')
            self.out_dir = out_dir.joinpath("net{}-{}"
                                            .format(identifier, time))
        print("Out directory: {}".format(self.out_dir))

    def _data_pipeline(self, params):
        # TODO: consider moving datasets
        train_dataset = self._loader.read_dataset(params['train_filepath'])
        val_dataset = self._loader.read_dataset(params['validate_filepath'])
        self.train_iter = train_dataset.shuffle(1000)\
                                       .batch(params['batch_size'])\
                                       .make_initializable_iterator()
        self.val_iter = val_dataset.batch(params['batch_size'])\
                                   .make_initializable_iterator()

    def _build_network(self, net_params):
        # Get input tensors.
        # one_hot_chars.shape = batch_size x max_chars x chars_in_vocab
        self.iter_handle, one_hot_chars, one_hot_labels\
            = self._loader.handle_to_input_tensors()
        labels = tf.math.argmax(one_hot_labels, 1)
        print("Shape of one_hot_chars: {}, labels: {}"
              .format(one_hot_chars.shape, labels.shape))

        # Create 1d convolutional layers.
        with tf.name_scope("conv"):
            tensor = one_hot_chars
            for conv in net_params['convolutional']:
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
            for dense in net_params['dense']:
                tensor = tf.layers.dense(inputs=tensor,
                                         units=dense['units'],
                                         activation=tf.nn.relu)
                print("Shape of tensor after dense: {}".format(tensor.shape))

        with tf.name_scope("output"):
            # Compute logits (1 per class).
            logits = tf.layers.dense(tensor, len(self._loader.vocab) + 1,
                                     activation=None)
            predictions = tf.argmax(logits, 1)
            self.outputs = {'logits': logits,
                            'predictions': predictions}

        print("Shape of labels: {}, logits: {}"
              .format(labels.shape, logits.shape))
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        # Create training op.
        self.learning_rate = tf.placeholder(tf.float32, shape=(),
                                            name='learning_rate')
        # print_outs = tf.print("outputs:", self.outputs, ", pred_shape:",
        #                       tf.shape(self.outputs['predictions']),
        #                       output_stream=sys.stdout, summarize=-1)
        # with tf.control_dependencies([print_outs]):
        self.train_op = tf.train.AdagradOptimizer(self.learning_rate)\
                                .minimize(loss, tf.train.get_global_step())

        with tf.name_scope("metrics"):
            self.metric_vals = {}
            self.metric_ops = {}
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

        # print_op = tf.print("labels: ", labels, "outputs:", self.outputs,
        #                     output_stream=sys.stdout)
        # Write summary.
        with tf.variable_scope("logging"): 
            # tf.control_dependencies([print_op]):  # noqa: E127
            for name, metric in self.metric_vals.items():
                tf.summary.scalar(name, metric)
            self.summary = tf.summary.merge_all()

        total_params = reduce(
            op.add,
            (reduce(op.mul, (dim.value for dim in var.get_shape()), 1)
             for var in tf.trainable_variables()),
            0)
        print("Total number of trainable parameters: {}".format(total_params))

    def _add_metric(self, name, metric):
        metric_value, metric_op = metric
        self.metric_vals[name] = metric_value
        self.metric_ops[name] = metric_op

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Free resources used by the network"""
        self._sess.close()

    def train_step(self, handle, learning_rate, writer=None) -> None:
        """TODO: train_step docstring"""
        _, metrics, summary, step = self._sess.run(
            (self.train_op, self.metric_ops,
             self.summary, tf.train.get_global_step()),
            feed_dict={self.iter_handle: handle,
                       self.learning_rate: learning_rate})
        if writer:
            writer.add_summary(summary, step)
        return metrics

    def run_epoch(self, runner: Callable[[], Mapping[str, float]])\
            -> Mapping[str, float]:
        """Runs a single epoch where the """
        self._sess.run(tf.local_variables_initializer())
        while True:
            try:
                runner()
            except tf.errors.OutOfRangeError:
                break
        return self._sess.run(self.metric_vals)

    def run(self, num_epochs, learning_rate, run_name=None):
        """TODO: train docstring"""
        run_dir = self.out_dir.joinpath("summaries", run_name or "unnamed")
        train_dir = run_dir.joinpath("train")
        val_dir = run_dir.joinpath("validate")
        with tf.summary.FileWriter(train_dir, self._sess.graph) as writer,\
             tf.summary.FileWriter(val_dir) as val_writer:  # noqa: E127
            handle = self._sess.run(self.train_iter.string_handle())
            for epoch in range(num_epochs):
                self._sess.run(self.train_iter.initializer)
                # pylint: disable=cell-var-from-loop
                metrics = self.run_epoch(lambda: self.train_step(
                    handle,
                    learning_rate(epoch)))
                writer.add_summary(self._sess.run(self.summary), epoch)
                print("Test: {}".format(self.test(val_writer, epoch)))
                yield metrics

    def test(self, writer=None, epoch=None) -> Mapping[str, float]:
        """TODO: test docstring"""
        self._sess.run(self.val_iter.initializer)
        handle = self._sess.run(self.val_iter.string_handle())
        metrics = self.run_epoch(lambda: self._sess.run(
            self.metric_ops,
            feed_dict={self.iter_handle: handle}))
        if writer and epoch is not None:
            writer.add_summary(self._sess.run(self.summary), epoch)
        return metrics

    # TODO: rewrite with iterator handles
    # def predict(self, features) -> np.ndarray:
    #     """TODO: predict docstring"""
    #     outputs = self._sess.run(self.outputs,
    #                              feed_dict={self.features: features})
    #     return {'class_ids': outputs['predictions'][:, tf.newaxis],
    #             'probabilities': tf.nn.softmax(outputs['logits']),
    #             'logits': outputs['logits']}

    def save_checkpoint(self, max_num_checkpoints) -> None:
        """Saves model with trained parameters"""
        checkpoint_dir = self.out_dir.joinpath("checkpoints")
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
        checkpoint_prefix = checkpoint_dir.joinpath("model")
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=max_num_checkpoints)
        saver.save(self._sess, checkpoint_prefix,
                   global_step=tf.train.get_global_step().eval())

    def restore_checkpoint(self, checkpoint: str) -> None:
        """Reads saved model from file"""
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self._sess, checkpoint)
