'''
Derivative of a function:
    Overview of automatic differentiation
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

x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# 元の入力テンソル x に対する z の微分
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0
        print('dz_dx[{0}][{1}] = {2}'.format(i, j, dz_dx[i][j].numpy()))

x = tf.linspace(-2 * pi, 2 * pi, 100)  # 100 points between -2π and +2π

with tf.GradientTape() as g:
    g.watch(x)
    y = tf.math.square(tf.math.sin(x))

plt.plot(x, g.gradient(y, x), label="first derivative")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.show()

print(
    '--------------------------------------------------------------------------\n'
    '  Case where there are two places where you want to take the derivative   \n'
    '--------------------------------------------------------------------------\n'
)


def get_derivative(inputs):
    x = tf.Variable(np.array([inputs, inputs], np.float32))  # gadientを取るためにVariableとする
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        y1 = x ** 2
        y2 = tf.math.log(y1)
    print("dy1/dx =", tape1.gradient(y1, x))
    print("dy2/dx =", tape2.gradient(y2, x))


for i in range(1, 4):
    print("i = ", i)
    get_derivative(i)

print(
    '--------------------------------------------------------------------------\n'
    '      Take a partial derivative of the second or higher order             \n'
    '--------------------------------------------------------------------------\n'
)
dx = []
ddx = []


def get_higher_derivative(input):
    x = tf.Variable(input)
    with tf.GradientTape() as tap1:
        with tf.GradientTape() as tap2:
            y = tf.math.square(tf.math.sin(x))
        dy_dx = tap2.gradient(y, x)
    d2y_dx2 = tap1.gradient(dy_dx, x)

    dx.append(dy_dx)
    ddx.append(d2y_dx2)


for tmp in x:
    get_higher_derivative(tmp)

'''
-----------------------------------------------------------
grad_f returns a list of derivatives of f with respect to its arguments.
f () has a single argument, so
grad_f returns a list with a single element.
----------------------------------------------------------
'''

plt.plot(x, dx, label="dy_dx derivative")
plt.plot(x, ddx, label="d2y_dx2 derivative")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.show()

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished         Derivative_of_a_function.py                             ({0})             \n'.format(
        date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()
