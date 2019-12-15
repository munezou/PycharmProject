from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
---------------------------------------------------------------------------------------------
TensorBoard Scalars: Logging training metrics in Keras

Overview)
Machine learning invariably involves understanding key metrics such as loss and how they change as training progresses. 
These metrics can help you understand if you're overfitting, for example, or if you're unnecessarily training for too long. 
You may want to compare these metrics across different training runs to help debug and improve your model.

TensorBoard's Scalars Dashboard allows you to visualize these metrics using a simple API with very little effort. 
This tutorial presents very basic examples to help you learn how to use these APIs with TensorBoard when developing your Keras model. 
You will learn how to use the Keras TensorBoard callback and TensorFlow Summary APIs to visualize default and custom scalars.
------------------------------------------------------------------------------------------------
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

data_size = 1000
# 80% of the data is for training.
train_pct = 0.8

train_size = int(data_size * train_pct)

# Create some input data between -1 and 1 and randomize it.
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)

# Generate the output data.
# y = 0.5x + 2 + noise
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size, ))

# Split into test and train pairs.
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# confirm the data
plt.figure(figsize=(8, 6))
plt.title("y = 0.5x + 2: raw data")
plt.scatter(x_train, y_train, c='red', label="train data")
plt.scatter(x_test, y_test, c='blue', label="test data")
plt.grid(True)
plt.legend()
plt.show()

logdir = os.path.join(PROJECT_ROOT_DIR, "logs", "scalars", datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1),
    keras.layers.Dense(1),
])

model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.2),
)

print("Training ... With default parameters, this takes less than 10 seconds.")
training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)

print("Average test loss: ", np.average(training_history.history['loss']))

print('model_predict([60, 25, 2]) = \n{0}\n'.format(model.predict([60, 25, 2])))

logdir = os.path.join(PROJECT_ROOT_DIR, "logs", "scalars", datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(os.path.join(logdir, "metrics"))
file_writer.set_as_default()

def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.2
    
    if epoch > 10:
        learning_rate = 0.02
    if epoch > 20:
        learning_rate = 0.01
    if epoch > 50:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = keras.models.Sequential([
                                keras.layers.Dense(16, input_dim=1),
                                keras.layers.Dense(1),
                                ])

model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(),
)

training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, lr_callback],
)

print("Average test loss: ", np.average(training_history.history['loss']))

print('model_predict([60, 25, 2]) = \n{0}\n'.format(model.predict([60, 25, 2])))