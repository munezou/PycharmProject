'''
tf.keras.optimizers.SGD:
    Stochastic gradient descent and momentum optimizer.
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

# d(loss)/d(var1) = var1
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
var = tf.Variable(1.0)
loss = lambda: (var ** 2) / 2.0
step_count = opt.minimize(loss, [var]).numpy()
# Step is `-learning_rate*grad`

print('var.numpy() = {0}\n'.format(var.numpy()))

opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
var = tf.Variable(1.0)
val0 = var.value()
loss = lambda: (var ** 2) / 2.0  # d(loss)/d(var1) = var1

# First step is `-learning_rate*grad`
step_count = opt.minimize(loss, [var]).numpy()
val1 = var.value()
print('(val0 - val1).numpy() = {0}\n'.format((val0 - val1).numpy()))

# On later steps, step-size increases because of momentum
step_count = opt.minimize(loss, [var]).numpy()
val2 = var.value()
print('(val1 - val2).numpy() = {0}\n'.format((val1 - val2).numpy()))

x = tf.Variable(1.0)
opt = tf.keras.optimizers.SGD(lr=0.1)

@tf.function
def step(i):
    with tf.GradientTape() as tape:
        square_x = x ** 2
        L = square_x

    tf.print('i = {0} / 29'.format(i))
    grad = tape.gradient(L, x)
    tf.print('grad =')
    tf.print(grad)
    opt.apply_gradients([(grad, x)])
    tf.print('x =')
    tf.print(x)
    tf.print()

for i in range(30):
    step(i)

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished         optimizers_SGD.py                              ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()
