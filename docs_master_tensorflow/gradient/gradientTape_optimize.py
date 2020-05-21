'''
GradientTape & Optimize:
    Use a gradient tape to find the optimum.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import io
import datetime
from packaging import version
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import tensorflow as tf

print(__doc__)

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

def train_step(x_input, y_input, a, ratio_learn):
    with tf.GradientTape() as tape:
        loss = tf.square(tf.math.multiply(x_input, a) - y_input)

    dloss_da = tf.gradients(loss, a)
    optimizer = tf.keras.optimizers.SGD(learning_rate=ratio_learn)
    optimizer.apply_gradients(zip(dloss_da, a))

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)




date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished         gradientTape_optimize.py                             ({0})             \n'.format(
        date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()
