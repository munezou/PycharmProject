"""
# Text Distances
#----------------------------------
#
# This function illustrates how to use
# the Levenstein distance (edit distance)
# in TensorFlow.
"""

# import required libraries
import os
import datetime
from packaging import version
import tensorflow as tf
from tensorflow.python.framework import ops

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

tf.compat.v1.disable_eager_execution()

ops.reset_default_graph()

sess = tf.compat.v1.Session()

# ----------------------------------
# First compute the edit distance between 'bear' and 'beers'
hypothesis = list('bear')
truth = list('beers')

h1 = tf.SparseTensor(
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
    hypothesis,
    [1, 1, 1]
)

t1 = tf.SparseTensor(
    [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 3], [0, 0, 4]],
    truth,
    [1, 1, 1]
)

print('tf.edit_distance(h1, t1, normalize=False) = {0}\n'.format(sess.run(tf.edit_distance(h1, t1, normalize=False))))

# ----------------------------------
# Compute the edit distance between ('bear','beer') and 'beers':
hypothesis2 = list('bearbeer')
truth2 = list('beersbeers')

h2 = tf.SparseTensor(
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]],
    hypothesis2,
    [1, 2, 4]
)

t2 = tf.SparseTensor(
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4]],
    truth2,
    [1, 2, 5]
)

print('tf.edit_distance(h2, t2, normalize=True) = {0}\n'.format(sess.run(tf.edit_distance(h2, t2, normalize=True))))

# ----------------------------------
# Now compute distance between four words and 'beers' more efficiently with sparse tensors:
hypothesis_words = ['bear', 'bar', 'tensor', 'flow']
truth_word = ['beers']

num_h_words = len(hypothesis_words)
h_indices = [[xi, 0, yi] for xi, x in enumerate(hypothesis_words) for yi, y in enumerate(x)]
h_chars = list(''.join(hypothesis_words))

h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words, 1, 1])

truth_word_vec = truth_word * num_h_words
t_indices = [[xi, 0, yi] for xi, x in enumerate(truth_word_vec) for yi, y in enumerate(x)]
t_chars = list(''.join(truth_word_vec))

t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words, 1, 1])

print('tf.edit_distance(h3, t3, normalize=True) = \n{0}\n'.format(sess.run(tf.edit_distance(h3, t3, normalize=True))))

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished         multiclass_svm_v2.py                         ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print()
print()
print()
