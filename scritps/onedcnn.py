"""TODO"""
from datetime import datetime
from functools import reduce
import operator as op
from pathlib import Path
from typing import Callable, Mapping, Optional
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
    metrics: Mapping[str, tf.Tensor]
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
        identifier = hash(repr((vocab, net_params)))
        for subdir in out_dir.iter_dir():
            if subdir.startswith("net{}".format(identifier)):
                self.out_dir = out_dir.joinpath(out_dir, subdir)
                return
        time = datetime.now().strftime('%Y-%m-%d-%H:%M')
        self.out_dir = out_dir.joinpath("net{}-{}".format(identifier, time))

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

        with tf.name_scope("metrics"):
            print("Shape of labels: {}, logits: {}"
                  .format(labels.shape, logits.shape))
            loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
            correct = tf.equal(predictions, labels)
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))
            useful = tf.logical_and(tf.not_equal(one_hot_labels[:, -1], 1),
                                    correct)
            real_accuracy = tf.reduce_mean(tf.cast(useful, "float"),
                                           name="real_accuracy")
            self.metrics = {'real_accuracy': real_accuracy,
                            'accuracy': accuracy,
                            'loss': loss}

        # print_op = tf.print("labels: ", labels, "outputs:", self.outputs,
        #                     output_stream=sys.stdout)
        # Write summary.
        with tf.variable_scope("logging"):
            # tf.control_dependencies([print_op]):  # noqa: E127
            for name, metric in self.metrics.items():
                tf.summary.scalar(name, metric)
            self.summary = tf.summary.merge_all()

        # Create training op.
        self.learning_rate = tf.placeholder(tf.float32, shape=(),
                                            name='learning_rate')
        # print_outs = tf.print("outputs:", self.outputs, ", pred_shape:",
        #                       tf.shape(self.outputs['predictions']),
        #                       output_stream=sys.stdout, summarize=-1)
        # with tf.control_dependencies([print_outs]):
        self.train_op = tf.train.AdagradOptimizer(self.learning_rate)\
                                .minimize(loss, tf.train.get_global_step())

        total_params = reduce(
            op.add,
            (reduce(op.mul, (dim.value for dim in var.get_shape()), 1)
             for var in tf.trainable_variables()),
            0)
        print("Total number of trainable parameters: {}".format(total_params))

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
            (self.train_op, self.metrics,
             self.summary, tf.train.get_global_step()),
            feed_dict={self.iter_handle: handle,
                       self.learning_rate: learning_rate})
        if writer:
            writer.add_summary(summary, step)
        return metrics

    def run_epoch(self, runner: Callable[[], Mapping[str, float]])\
            -> Mapping[str, float]:
        """Runs a single epoch where the """
        metrics = dict((key, 0) for key in self.metrics)
        num_batches = 0
        while True:
            try:
                batch_metrics = runner()
                num_batches += 1
                for key, val in batch_metrics.items():
                    metrics[key] += val
            except tf.errors.OutOfRangeError:
                for key in metrics:
                    metrics[key] /= num_batches
                return metrics

    def train(self, num_epochs, learning_rate):
        """TODO: train docstring"""
        # TODO: fix metrics mess
        train_sum_dir = self.out_dir.joinpath("summaries", "train")
        val_sum_dir = self.out_dir.joinpath("summaries", "validate")
        with tf.summary.FileWriter(train_sum_dir, self._sess.graph) as writer:
            handle = self._sess.run(self.train_iter.string_handle())
            for epoch in range(num_epochs):
                self._sess.run(self.train_iter.initializer)
                metrics = self.run_epoch(lambda: self.train_step(
                    handle,
                    learning_rate(epoch),  # pylint: disable=cell-var-from-loop
                    writer))
                writer.flush()
                print("Test: {}".format(self.test(val_sum_dir)))
                yield metrics

    def test(self, summary_dir: Optional[Path] = None) -> Mapping[str, float]:
        """TODO: test docstring"""
        with tf.summary.FileWriter(summary_dir, self._sess.graph) as writer:
            self._sess.run(self.val_iter.initializer)
            handle = self._sess.run(self.val_iter.string_handle())
            metrics = self.run_epoch(lambda: self._sess.run(
                self.metrics,
                feed_dict={self.iter_handle: handle}))
            # writer.add_summary(metrics, step)
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
