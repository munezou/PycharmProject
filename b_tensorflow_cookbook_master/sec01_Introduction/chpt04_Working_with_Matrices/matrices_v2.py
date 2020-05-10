# Matrices and Matrix Operations
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow
import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Declaring matrices
sess = tf.compat.v1.Session()

# Declaring matrices

# Identity matrix
identity_matrix = tf.linalg.tensor_diag([1.0,1.0,1.0])
print(sess.run(identity_matrix))

# 2x3 random norm matrix
A = tf.random.truncated_normal([2,3])
print(sess.run(A))

# 2x3 constant matrix
B = tf.fill([2,3], 5.0)
print(sess.run(B))

# 3x2 random uniform matrix
C = tf.random.uniform([3,2])
print(sess.run(C))  # Note that we are reinitializing, hence the new random variables

# Create matrix from np array
D = tf.convert_to_tensor(value=np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))

# Matrix addition/subtraction
print(sess.run(A+B))
print(sess.run(B-B))

# Matrix Multiplication
print(sess.run(tf.matmul(B, identity_matrix)))

# Matrix Transpose
print(sess.run(tf.transpose(a=C))) # Again, new random variables

# Matrix Determinant
print(sess.run(tf.linalg.det(D)))

# Matrix Inverse
print(sess.run(tf.linalg.inv(D)))

# Cholesky Decomposition
print(sess.run(tf.linalg.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors
print(sess.run(tf.linalg.eigh(D)))