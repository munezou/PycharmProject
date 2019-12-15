from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
---------------------------------------------------------------------------------------
Examining the TensorFlow Graph


Overview
TensorBoard’s Graphs dashboard is a powerful tool for examining your TensorFlow model. 
You can quickly view a conceptual graph of your model’s structure and ensure it matches your intended design. 
You can also view a op-level graph to understand how TensorFlow understands your program. 
Examining the op-level graph can give you insight as to how to change your model. 
For example, you can redesign your model if training is progressing slower than expected.

This tutorial presents a quick overview of how to generate graph diagnostic data and visualize it in TensorBoard’s Graphs dashboard. 
You’ll define and train a simple Keras Sequential model for the Fashion-MNIST dataset and learn how to log and examine your model graphs. 
You will also use a tracing API to generate graph data for functions created using the new tf.function annotation.
--------------------------------------------------------------------------------------
'''
print(__doc__)

# common library
import subprocess
import os
import platform
import shutil
from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

# Clear any logs from previous runs
pathLogs = os.path.join(PROJECT_ROOT_DIR, "logs")

try:
    if pf == 'Linux':
        runcmd = subprocess.call(["rm", "-rf", pathLogs])
    elif pf == 'Windows':
        runcmd = shutil.rmtree(pathLogs)
    
    print(runcmd)
    pass
except Exception as ex:
    print(ex)
    pass
finally:
    pass

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Define a Keras model                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

# In this example, the classifier is a simple four-layer Sequential model.
# Define the model.
model = keras.models.Sequential ([
                                keras.layers.Flatten(input_shape=(28, 28)),
                                keras.layers.Dense(32, activation='relu'),
                                keras.layers.Dropout(0.2),
                                keras.layers.Dense(10, activation='softmax')
                                ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Download and prepare the training data.
(train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Train the model and log data                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------
Before training, define the Keras TensorBoard callback, specifying the log directory. 
By passing this callback to Model.fit(), you ensure that graph data is logged for visualization in TensorBoard.
-------------------------------------------------------------------------------------
'''
# Define the Keras TensorBoard callback.
logdir = os.path.join(PROJECT_ROOT_DIR, "logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model.
model.fit   (
            train_images,
            train_labels, 
            batch_size=64,
            epochs=5, 
            callbacks=[tensorboard_callback]
            )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Op-level graph                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------------
By default, TensorBoard displays the op-level graph. (On the left, you can see the “Default” tag selected.) 
Note that the graph is inverted; data flows from bottom to top, so it’s upside down compared to the code. 
However, you can see that the graph closely matches the Keras model definition, with extra edges to other computation nodes.

Graphs are often very large, so you can manipulate the graph visualization:

* Scroll to zoom in and out
* Drag to pan
* Double clicking toggles node expansion (a node can be a container for other nodes)

You can also see metadata by clicking on a node. This allows you to see inputs, outputs, shapes and other details.
-------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Graphs of tf.functions                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------------
The examples so far have described graphs of Keras models, 
where the graphs have been created by defining Keras layers and calling Model.fit().

You may encounter a situation where you need to use the tf.function annotation to "autograph", i.e., transform, 
a Python computation function into a high-performance TensorFlow graph. 
For these situations, you use TensorFlow Summary Trace API to log autographed functions for visualization in TensorBoard.

To use the Summary Trace API:

Define and annotate a function with tf.function
Use tf.summary.trace_on() immediately before your function call site.
Add profile information (memory, CPU time) to graph by passing profiler=True
With a Summary file writer, call tf.summary.trace_export() to save the log data
You can then use TensorBoard to see how your function behaves.
---------------------------------------------------------------------------------------------------
'''
# The function to be traced.
@tf.function
def my_func(x, y):
    # A simple hand-rolled layer.
    return tf.nn.relu(tf.matmul(x, y))

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(PROJECT_ROOT_DIR, "logs", "func", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = my_func(x, y)
with writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir=logdir)

'''
----------------------------------------------------------------------------------------------------
You can now see the structure of your function as understood by TensorBoard. 
Click on the "Profile" radiobutton to see CPU and memory statistics.
----------------------------------------------------------------------------------------------------
'''
