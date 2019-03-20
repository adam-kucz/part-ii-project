#!/usr/bin/env python
import tensorflow as tf
from deeplearn.util import sortdict

hello = tf.constant("hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))

test_dict = {"hi": 0, "abc": 1, "world": 2, "hello": 3}
print(sortdict(test_dict))
