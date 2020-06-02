'''
# Implementing Gates
#----------------------------------
#
# This function shows how to implement
# various gates in TensorFlow
#
# One gate will be one operation with
# a variable and a placeholder.
# We will ask TensorFlow to change the
# variable based on our loss function
'''

# import required libraries
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import tensorflow as tf
from tensorflow.python.framework import ops

print(__doc__)

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

#----------------------------------
# Create a multiplication gate:
#   f(x) = a * x
#
#  a --
#      |
#      |---- (multiply) --> output
#  x --|
#

a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.compat.v1.placeholder(dtype=tf.float32)

multiplication = tf.multiply(a, x_data)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.subtract(multiplication, 50.))

# Initialize variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.compat.v1.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Run loop across gate
print('Optimizing a Multiplication Gate Output to 50.')
for _ in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
    
'''
Create a nested gate:
   f(x) = a * x + b

  a --
      |
      |-- (multiply)--
  x --|              |
                     |-- (add) --> output
                 b --|

'''

# Start a New Graph Session
ops.reset_default_graph()
sess = tf.compat.v1.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.compat.v1.placeholder(dtype=tf.float32)

two_gate = tf.add(tf.multiply(a, x_data), b)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.subtract(two_gate, 50.))

# Initialize variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.compat.v1.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Run loop across gate
print('\nOptimizing Two Gate Output to 50.')
for _ in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val, b_val = (sess.run(a), sess.run(b))
    two_gate_output = sess.run(two_gate, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished      gates_v2.py                          ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()