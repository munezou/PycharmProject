from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
---------------------------------------------------------------------
Get started with TensorBoard

Tensorboard

In machine learning, 
to improve something you often need to be able to measure it. 
TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. 
It enables tracking experiment metrics like loss and accuracy, 
visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.

This quickstart will show how to quickly get started with TensorBoard. 
The remaining guides in this website provide more details on specific capabilities, many of which are not included here.
--------------------------------------------------------------------
'''
print(__doc__)


from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

import numpy as np

import os

import matplotlib.pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

'''
---------------------------------------------------------------------
Using the MNIST dataset as the example, 
normalize the data and write a function 
that creates a simple Keras model for classifying the images into 10 classes.
---------------------------------------------------------------------
'''
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential   ([
                                        tf.keras.layers.Flatten(input_shape=(28, 28)),
                                        tf.keras.layers.Dense(512, activation='relu'),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(10, activation='softmax')
                                        ])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '          Using TensorBoard with Keras Model.fit()                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------
When training with Keras's Model.fit(), 
adding the tf.keras.callback.TensorBoard callback ensures that logs are created and stored. 
Additionally, enable histogram computation every epoch with histogram_freq=1 (this is off by default)
---------------------------------------------------------------------
'''
# Place the logs in a timestamped subdirectory to allow easy selection of different training runs.
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logdir = os.path.join(PROJECT_ROOT_DIR, "logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

model.fit(x=x_train, y=y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

'''
--------------------------------------------------------------------
Start TensorBoard through the command line or within a notebook experience. 
The two interfaces are generally the same. 
In notebooks, use the %tensorboard line magic.
On the command line, run the same command without "%".
-------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '          Using TensorBoard with other methods                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------
When training with methods such as tf.GradientTape(), 
use tf.summary to log the required information.

Use the same dataset as above, 
but convert it to tf.data.Dataset to take advantage of batching capabilities:
---------------------------------------------------------------------------------
'''
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(60000).batch(64)
test_dataset = test_dataset.batch(64)

'''
--------------------------------------------------------------------------------
The training code follows the advanced quickstart tutorial, 
but shows how to log metrics to TensorBoard. 
Choose loss and optimizer:
-------------------------------------------------------------------------------
'''
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

'''
-------------------------------------------------------------------------------
Create stateful metrics that can be used to accumulate values during training 
and logged at any point:
-------------------------------------------------------------------------------
'''
# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

'''
-----------------------------------------------------------------------------
Define the training and test functions:
-----------------------------------------------------------------------------
'''
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_train, predictions)

def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)
    test_accuracy(y_test, predictions)

'''
------------------------------------------------------------------------------
Set up summary writers to write the summaries to disk in a different logs directory:
------------------------------------------------------------------------------
'''
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_dir = os.path.join(PROJECT_ROOT_DIR, "logs", "gradient_tape", current_time, "train")
test_log_dir = os.path.join(PROJECT_ROOT_DIR, "logs", "gradient_tape", current_time, "test")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

'''
--------------------------------------------------------------------------------
Start training. 
Use tf.summary.scalar() to log metrics (loss and accuracy) 
during training/testing within the scope of the summary writers to write the summaries to disk. 
You have control over which metrics to log and how often to do it. 
Other tf.summary functions enable logging other types of data.
--------------------------------------------------------------------------------
'''
model = create_model() # reset our model

EPOCHS = 5

for epoch in range(EPOCHS):
    for (x_train, y_train) in train_dataset:
        train_step(model, optimizer, x_train, y_train)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for (x_test, y_test) in test_dataset:
        test_step(model, x_test, y_test)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

'''
--------------------------------------------------------------------------------
Open TensorBoard again, this time pointing it at the new log directory. 
We could have also started TensorBoard to monitor training while it progresses.
--------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------
That's it! 
You have now seen how to use TensorBoard both through the Keras callback 
and through tf.summary for more custom scenarios.
-------------------------------------------------------------------------------
'''
