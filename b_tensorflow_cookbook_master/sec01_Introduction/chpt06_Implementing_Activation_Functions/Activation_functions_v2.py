# Activation Functions
# ----------------------------------
#
# This function introduces activation
# functions in TensorFlow

# Implementing Activation Functions
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from packaging import version
from tensorflow.python.framework import ops

tf.compat.v1.disable_eager_execution()
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

# Open graph session
sess = tf.compat.v1.Session()

# X range
x_vals = np.linspace(start=-10., stop=10., num=100)

# ReLU activation
print('tf.nn.relu([-3., 3., 10.]) = {0}\n'.format(sess.run(tf.nn.relu([-3., 3., 10.]))))
y_relu = sess.run(tf.nn.relu(x_vals))

# ReLU-6 activation
print('tf.nn.relu6([-3., 3., 10.]) = {0}\n'.format(sess.run(tf.nn.relu6([-3., 3., 10.]))))
y_relu6 = sess.run(tf.nn.relu6(x_vals))

# Sigmoid activation
print('sess.run(tf.nn.sigmoid([-1., 0., 1.]) = {0}\n'.format(sess.run(tf.nn.sigmoid([-1., 0., 1.]))))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# Hyper Tangent activation
print('tf.nn.tanh([-1., 0., 1.]) = {0}\n'.format(sess.run(tf.nn.tanh([-1., 0., 1.]))))
y_tanh = sess.run(tf.nn.tanh(x_vals))

# Softsign activation
print('tf.nn.softsign([-1., 0., 1.]) = {0}\n'.format(sess.run(tf.nn.softsign([-1., 0., 1.]))))
y_softsign = sess.run(tf.nn.softsign(x_vals))

# Softplus activation
print('tf.nn.softplus([-1., 0., 1.]) = {0}\n'.format(sess.run(tf.nn.softplus([-1., 0., 1.]))))
y_softplus = sess.run(tf.nn.softplus(x_vals))

# Exponential linear activation
print('tf.nn.elu([-1., 0., 1.]) = {0}\n'.format(sess.run(tf.nn.elu([-1., 0., 1.]))))
y_elu = sess.run(tf.nn.elu(x_vals))

# Plot the different functions
plt.plot(x_vals, y_softplus, 'r--', label='Softplus', linewidth=2)
plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
plt.plot(x_vals, y_elu, 'k-', label='ExpLU', linewidth=0.5)
plt.ylim([-1.5,7])
plt.legend(loc='upper left')
plt.show()

plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
plt.plot(x_vals, y_softsign, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2,2])
plt.legend(loc='upper left')
plt.show()

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
    )

print(
    '       finished         Activation_function_v2.py                                  ({0})    \n'.format(date_today)
    )

print(
    '------------------------------------------------------------------------------------------------------\n'
    )

print()
print()
print()