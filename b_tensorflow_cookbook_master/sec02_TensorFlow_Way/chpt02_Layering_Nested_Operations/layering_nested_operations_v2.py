# Layering Nested Operations

import os
import datetime
from packaging import version
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

tf.compat.v1.disable_eager_execution()

ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

# Start a graph session
sess = tf.compat.v1.Session()

# Create the data and variables
my_array = np.array(
    [[1., 3., 5., 7., 9.],
   [-2., 0., 2., 4., 6.],
   [-6., -3., 0., 3., 6.]]
)

x_vals = np.array([my_array, my_array + 1])
x_data = tf.compat.v1.placeholder(tf.float32, shape=(3, 5))

# Constants for matrix multiplication:
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# Create our multiple operations
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

# Now feed data through placeholder and print results
for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))

# View the tensorboard graph by running the following code and then
#    going to the terminal and typing:
#    $ tensorboard --logdir=tensorboard_logs
merged = tf.compat.v1.summary.merge_all()
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')

my_writer = tf.compat.v1.summary.FileWriter('tensorboard_logs/', sess.graph)

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished         layering_nested_operations_v2.py                           ({0})    \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print()
print()
print()
