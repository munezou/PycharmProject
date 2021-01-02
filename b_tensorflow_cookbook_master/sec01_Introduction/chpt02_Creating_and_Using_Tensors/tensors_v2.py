# Tensors
#----------------------------------
#
# This function introduces various ways to create
# tensors in TensorFlow
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

# Introduce tensors in tf

# Get graph handle
sess = tf.compat.v1.Session()

my_tensor = tf.zeros([1, 20])

# Declare a variable
my_var = tf.Variable(tf.zeros([1, 20]))

# Different kinds of variables
row_dim = 2
col_dim = 3

# Zero initialized variable
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))

# One initialized variable
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))

# shaped like other variable
sess.run(zero_var.initializer)
sess.run(ones_var.initializer)
zero_similar = tf.Variable(tf.zeros_like(zero_var))
ones_similar = tf.Variable(tf.ones_like(ones_var))

sess.run(ones_similar.initializer)
sess.run(zero_similar.initializer)

# Fill shape with a constant
fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))

# Create a variable from a constant
const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
# This can also be used to fill an array:
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))

# Sequence generation
linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end

sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end

# Random Numbers

# Random Normal
rnorm_var = tf.random.normal([row_dim, col_dim], mean=0.0, stddev=1.0)

# Add summaries to tensorboard
merged = tf.compat.v1.summary.merge_all()

# Initialize graph writer:
writer = tf.compat.v1.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

# Initialize operation
initialize_op = tf.compat.v1.global_variables_initializer()

# Run initialization of variable
sess.run(initialize_op)

date_today = datetime.date.today()

print(
	'------------------------------------------------------------------------------------------------------\n'
)

print(
	'       finished         tensors_v2.py                                  ({0})             \n'.format(date_today)
)

print(
	'------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()