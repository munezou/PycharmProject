# -*- coding: utf-8 -*-
# Parallelizing TensorFlow
#----------------------------------
#
# We will show how to use TensorFlow distributed
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# We will setup a local cluster (on localhost)

# Cluster for 2 local workers (tasks 0 and 1):
cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})
# Server definition:
server = tf.distribute.Server(cluster, job_name="local", task_index=0)
server = tf.distribute.Server(cluster, job_name="local", task_index=1)
# Finish and add
# server.join()

# Have each worker do a task
# Worker 0 : create matrices
# Worker 1 : calculate sum of all elements
mat_dim = 25
matrix_list = {}

with tf.device('/job:local/task:0'):
    for i in range(0, 2):
        m_label = 'm_{}'.format(i)
        matrix_list[m_label] = tf.random.normal([mat_dim, mat_dim])

# Have each worker calculate the Cholesky Decomposition
sum_outs = {}
with tf.device('/job:local/task:1'):
    for i in range(0, 2):
        A = matrix_list['m_{}'.format(i)]
        sum_outs['m_{}'.format(i)] = tf.reduce_sum(input_tensor=A)

    # Sum all the cholesky decompositions
    summed_out = tf.add_n(list(sum_outs.values()))

with tf.compat.v1.Session(server.target) as sess:
    result = sess.run(summed_out)
    print('Summed Values:{}'.format(result))
