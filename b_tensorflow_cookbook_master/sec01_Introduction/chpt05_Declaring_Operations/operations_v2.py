# Operations
# ----------------------------------
#
# This function introduces various operations
# in TensorFlow

# Declaring Operations
import os
import datetime
import tensorflow as tf
from packaging import version
from tensorflow.python.framework import ops

tf.compat.v1.disable_eager_execution()

ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

# Open graph session
sess = tf.compat.v1.Session()

# div() vs truediv() vs floordiv()
print('tf.compat.v1.div(3, 4) = {0}'.format(sess.run(tf.compat.v1.div(3, 4))))
print('tf.truediv(3, 4) = {0}'.format(sess.run(tf.truediv(3, 4))))
print('tf.math.floordiv(3.0, 4.0) = {0}\n'.format(sess.run(tf.math.floordiv(3.0, 4.0))))

# Mod function
print('tf.math.floormod(22.0, 5.0) = {0}\n'.format(sess.run(tf.math.floormod(22.0, 5.0))))

# Cross Product
print(
    'tf.linalg.cross([1., 0., 0.], [0., 1., 0.]) = {0}\n'.format(sess.run(tf.linalg.cross([1., 0., 0.], [0., 1., 0.]))))

# Trig functions
print('tf.sin(3.1416) = {0}\n'.format(sess.run(tf.sin(3.1416))))
print('tf.cos(3.1416) = {0}\n'.format(sess.run(tf.cos(3.1416))))
print('tf.tan(3.1416/4.) = {0}\n'.format(sess.run(tf.tan(3.1416 / 4.))))

# Custom operation
test_nums = range(15)


def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
    return tf.subtract(3 * tf.square(x_val), x_val) + 10


print('custom_polynomial(11) = {0}\n'.format(sess.run(custom_polynomial(11))))

# What should we get with list comprehension
expected_output = [3 * x * x - x + 10 for x in test_nums]
print('expected_output = {0}\n'.format(expected_output))

# TensorFlow custom function output
for num in test_nums:
    print('custom_polynomial(num) = {0}\n'.format(sess.run(custom_polynomial(num))))

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished         operations_v2.py                                  ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()
