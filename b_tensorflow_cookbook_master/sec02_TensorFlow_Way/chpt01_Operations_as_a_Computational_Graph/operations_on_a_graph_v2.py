# Operations on a Computational Graph
import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Create graph
sess = tf.compat.v1.Session()

# Create tensors

# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.compat.v1.placeholder(tf.float32)
m_const = tf.constant(3.)

# Multiplication
my_product = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))

# View the tensorboard graph by running the following code and then
#    going to the terminal and typing:
#    $ tensorboard --logdir=tensorboard_logs
merged = tf.compat.v1.summary.merge_all()
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')

my_writer = tf.compat.v1.summary.FileWriter('tensorboard_logs/', sess.graph)
