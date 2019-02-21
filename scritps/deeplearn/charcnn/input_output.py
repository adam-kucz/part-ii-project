from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import tensorflow as tf

from .. import util as myutil

__all__ = ['IO']


class IO:
    def __init__(self,
                 out_dir: Path,
                 params: Mapping[str, Any],
                 metrics: Mapping[str, tf.Tensor],
                 sess: tf.Session):
        self.sess = sess
        self._set_out_dir(out_dir, params)
        self._make_summary(metrics)
        self._run_name: Optional[str] = None
        self._new_writers()

    def _make_summary(self, metrics):
        with tf.variable_scope("summary"):
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric)
            self._summary = tf.summary.merge_all()

    def _set_out_dir(self, out_dir: Path, net_params: Any):
        identifier = str(myutil.stable_hash(net_params).hex())
        subdirs = tuple(out_dir.glob("net{}-*".format(identifier)))
        if subdirs:
            self.out_dir = out_dir.joinpath(out_dir, subdirs[0])
        else:
            time = datetime.now().strftime('%Y-%m-%d-%H-%M')
            self.out_dir = out_dir.joinpath("net{}-{}"
                                            .format(identifier, time))
            self.out_dir.mkdir(parents=True)
            with self.out_dir.joinpath("specification").open('w') as specfile:
                specfile.write("Network {}\n\nParameters:\n"
                               .format(identifier))
                specfile.writelines("{}: {}\n".format(param, val)
                                    for param, val in net_params.items())
        print("Out directory: {}".format(self.out_dir))

    @property
    def summary(self):
        return self.sess.run(self._summary)

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, value):
        if value != self.run_name:
            self._run_name = value
            self.close()
            self._new_writers()

    def _new_writers(self):
        run_dir = self.out_dir.joinpath("summaries",
                                        self.run_name or "unnamed")
        train_dir = run_dir.joinpath("train")
        self.train_writer = tf.summary.FileWriter(train_dir,
                                                  self.sess.graph)
        val_dir = run_dir.joinpath("validate") 
        self.test_writer = tf.summary.FileWriter(val_dir,
                                                 self.sess.graph)

    def __enter__(self):
        return self

    def __exit(self, *_):
        self.close()

    def close(self):
        self.train_writer.close()
        self.test_writer.close()

    def write_train_summary(self, step: int):
        self.train_writer.add_summary(self.summary, step)

    def write_test_summary(self, step: Optional[int] = None):
        self.test_writer.add_summary(self.summary, step)

    def save_checkpoint(self, max_num_checkpoints: int):
        checkpoint_dir = self.out_dir.joinpath("checkpoints")
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
        checkpoint_prefix = checkpoint_dir.joinpath("model")
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=max_num_checkpoints)
        saver.save(self.sess, checkpoint_prefix,
                   global_step=tf.train.get_global_step())

    def restore_checkpoint(self, checkpoint: Optional[str] = None) -> None:
        if not checkpoint:
            checkpoint = tf.train.latest_checkpoint(
                self.out_dir.joinpath("checkpoints"))
            if not checkpoint:
                raise ValueError("No checkpoint to restore")
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, checkpoint)
