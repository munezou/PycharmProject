# Matrices and Matrix Operations
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow
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

# Declaring matrices
sess = tf.compat.v1.Session()

# Declaring matrices

# Identity matrix
identity_matrix = tf.linalg.tensor_diag([1.0, 1.0, 1.0])
print('identity_matrix = \n{0}\n'.format(sess.run(identity_matrix)))

# 2x3 random norm matrix
A = tf.random.truncated_normal([2, 3])
print('A = \n{0}\n'.format(sess.run(A)))

# 2x3 constant matrix
B = tf.fill([2, 3], 5.0)
print('B = \n{0}\n'.format(sess.run(B)))

# 3x2 random uniform matrix
C = tf.random.uniform([3, 2])
print('C = \n{0}\n'.format(sess.run(C))) # Note that we are reinitializing, hence the new random variables

# Create matrix from np array
D = tf.convert_to_tensor(value=np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print('D = \n{0}\n'.format(sess.run(D)))

# Matrix addition/subtraction
print('A+B = \n{0}\n'.format(sess.run(A+B)))
print('B-B = \n{0}\n'.format(sess.run(B-B)))

# Matrix Multiplication
print('B*identity_matrix = \n{0}\n'.format(sess.run(tf.matmul(B, identity_matrix))))

# Matrix Transpose
print('tf.transpose(a=C) = \n{0}\n'.format(sess.run(tf.transpose(a=C)))) # Again, new random variables

# Matrix Determinant
print('tf.linalg.det(D) = \n{0}\n'.format(sess.run(tf.linalg.det(D))))

# Matrix Inverse
print('tf.linalg.inv(D) = \n{0}\n'.format(sess.run(tf.linalg.inv(D))))

# Cholesky Decomposition
print('tf.linalg.cholesky(identity_matrix) = \n{0}\n'.format(sess.run(tf.linalg.cholesky(identity_matrix))))

# Eigenvalues and Eigenvectors
print('tf.linalg.eigh(D) = \n{0}\n'.format(sess.run(tf.linalg.eigh(D))))

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         matrices_v2.py                                  ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()