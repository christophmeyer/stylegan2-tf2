#import unittest

import tensorflow as tf

from model import Conv2DModulated
from tensorflow.python.framework import random_seed


class TestConv2DModulated(tf.test.TestCase):
    def setUp(self):
        self.conv2d_mod = Conv2DModulated(filters=10,
                                          kernel_size=3,
                                          lr_multiplier=1.0,
                                          gain=1.0,
                                          modulate=True,
                                          demodulate=True)
        random_seed.set_seed(42)

    def test_shape(self):
        self.dlatents_const = tf.zeros([2, 1, 10])
        example = tf.random.normal([1, 4, 4, 2])
        self.inputs = tf.tile(example, [2, 1, 1, 1])

        outputs = self.conv2d_mod(self.inputs, self.dlatents_const)
        self.assertAllClose(outputs[0], outputs[1])
        self.assertEqual(outputs.shape, tf.TensorShape([2, 4, 4, 10]))
