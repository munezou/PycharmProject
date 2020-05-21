# Placeholders
#----------------------------------
#
# This function introduces how to 
# use placeholders in TensorFlow
import os
import datetime
import numpy as np
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

# Using Placeholders
sess = tf.compat.v1.Session()

x = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

rand_array = np.random.rand(4, 4)

merged = tf.compat.v1.summary.merge_all()

writer = tf.compat.v1.summary.FileWriter("/tmp/variable_logs", sess.graph)

print(sess.run(y, feed_dict={x: rand_array}))

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         placeholder_v2.py                                  ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()