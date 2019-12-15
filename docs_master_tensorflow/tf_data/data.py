'''
----------------------------------------------------------------------------------------------
tf.data: Build TensorFlow input pipelines

overview)

The tf.data API enables you to build complex input pipelines from simple, reusable pieces. For example, 
the pipeline for an image model might aggregate data from files in a distributed file system, 
apply random perturbations to each image, and merge randomly selected images into a batch for training. 
The pipeline for a text model might involve extracting symbols from raw text data, 
converting them to embedding identifiers with a lookup table, and batching together sequences of different lengths. 
The tf.data API makes it possible to handle large amounts of data, 
read from different data formats, and perform complex transformations.

The tf.data API introduces a tf.data.Dataset abstraction that represents a sequence of elements, 
in which each element consists of one or more components. 
For example, in an image pipeline, an element might be a single training example, 
with a pair of tensor components representing the image and its label.

There are two distinct ways to create a dataset:

A data source constructs a Dataset from data stored in memory or in one or more files.

A data transformation constructs a dataset from one or more tf.data.Dataset objects.
----------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import platform
import shutil
import subprocess
from packaging import version
from PIL import Image
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

np.set_printoptions(precision=4)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Basic mechanics                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To create an input pipeline, you must start with a data source. 
For example, to construct a Dataset from data in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices(). 
Alternatively, if your input data is stored in a file in the recommended TFRecord format, you can use tf.data.TFRecordDataset().

Once you have a Dataset object, you can transform it into a new Dataset by chaining method calls on the tf.data.Dataset object. 
For example, you can apply per-element transformations such as Dataset.map(), 
and multi-element transformations such as Dataset.batch(). 
See the documentation for tf.data.Dataset for a complete list of transformations.

The Dataset object is a Python iterable. This makes it possible to consume its elements using a for loop:
----------------------------------------------------------------------------------------------------------------
'''
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
print('dataset = \n{0}\n'.format(dataset))

for elem in dataset:
  print('elem = {0}'.format(elem.numpy()))