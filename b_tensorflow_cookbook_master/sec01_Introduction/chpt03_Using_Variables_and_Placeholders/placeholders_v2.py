# Placeholders
#----------------------------------
#
# This function introduces how to 
# use placeholders in TensorFlow
import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Using Placeholders
sess = tf.compat.v1.Session()

x = tf.compat.v1.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

rand_array = np.random.rand(4, 4)

merged = tf.compat.v1.summary.merge_all()

writer = tf.compat.v1.summary.FileWriter("/tmp/variable_logs", sess.graph)

print(sess.run(y, feed_dict={x: rand_array}))