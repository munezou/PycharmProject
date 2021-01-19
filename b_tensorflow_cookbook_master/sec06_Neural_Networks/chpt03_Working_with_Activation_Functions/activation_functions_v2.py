"""
Combining Gates and Activation Functions

This function shows how to implement
various gates with activation functions
in TensorFlow

This function is an extension of the
prior gates, but with various activation
functions.
"""

# import required libraries
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

print(__doc__)

"""
--------------------------------------------
In casee of windows, os name is 'nt'.
In case of linux, os name is 'posix'.
--------------------------------------------
"""

if os.name == 'nt':
    print(
        '--------------------------------------------------------------------------\n'
        '                      cpu information                                     \n'
        '--------------------------------------------------------------------------\n'
    )
    # display the using cpu information
    for key, value in get_cpu_info().items():
        print("{0}: {1}".format(key, value))

    print()
    print()

# Display current path
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: {0}\n".format(tf.version.VERSION))
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

ops.reset_default_graph()

tf.compat.v1.disable_eager_execution()

# Start Graph Session
sess = tf.compat.v1.Session()
tf.compat.v1.set_random_seed(5)
np.random.seed(42)

batch_size = 50

a1 = tf.Variable(tf.random.normal(shape=[1, 1]))
b1 = tf.Variable(tf.random.uniform(shape=[1, 1]))
a2 = tf.Variable(tf.random.normal(shape=[1, 1]))
b2 = tf.Variable(tf.random.uniform(shape=[1, 1]))
x = np.random.normal(2, 0.1, 500)
x_data = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)

sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))

relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

# Declare the loss function as the difference between
# the output and a target value, 0.75.
loss1 = tf.reduce_mean(input_tensor=tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2 = tf.reduce_mean(input_tensor=tf.square(tf.subtract(relu_activation, 0.75)))

# Initialize variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.compat.v1.train.GradientDescentOptimizer(0.01)
train_step_sigmoid = my_opt.minimize(loss1)
train_step_relu = my_opt.minimize(loss2)

# Run loop across gate
print('\nOptimizing Sigmoid AND Relu Output to 0.75')
loss_vec_sigmoid = []
loss_vec_relu = []

activation_sigmoid = []
activation_relu = []

for i in range(500):
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = np.transpose([x[rand_indices]])
    sess.run(train_step_sigmoid, feed_dict={x_data: x_vals})
    sess.run(train_step_relu, feed_dict={x_data: x_vals})
    
    loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals}))
    loss_vec_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))    
    
    sigmoid_output = np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals}))
    relu_output = np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals}))

    activation_sigmoid.append(sigmoid_output)
    activation_relu.append(relu_output)
    
    if i % 50 == 0:
        print('sigmoid = ' + str(np.mean(sigmoid_output)) + ' relu = ' + str(np.mean(relu_output)))

# plot Activation function
plt.plot(activation_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(activation_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Activation Outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')
plt.show()

# Plot the loss
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished      activation_functions_v2.py                ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print()
print()
print()