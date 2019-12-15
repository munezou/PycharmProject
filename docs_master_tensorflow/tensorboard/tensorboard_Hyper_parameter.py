from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
---------------------------------------------------------------------------------------
Hyperparameter Tuning with the HParams Dashboard

Overview
When building machine learning models, you need to choose various hyperparameters, 
such as the dropout rate in a layer or the learning rate. 
These decisions impact model metrics, such as accuracy. 
Therefore, an important step in the machine learning workflow is to identify the best hyperparameters for your problem, 
which often involves experimentation. 
This process is known as "Hyperparameter Optimization" or "Hyperparameter Tuning".

The HParams dashboard in TensorBoard provides several tools 
to help with this process of identifying the best experiment or most promising sets of hyperparameters.

This tutorial will focus on the following steps:

1. Experiment setup and HParams summary
2. Adapt TensorFlow runs to log hyperparameters and metrics
3. Start runs and log them all under one parent directory
4. Visualize the results in TensorBoard's HParams dashboard
--------------------------------------------------------------------------------------
'''
print(__doc__)

# common library
import subprocess
from datetime import datetime
import io
import os
import sys
import platform
import shutil
from packaging import version

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

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

# Download the FashionMNIST dataset and scale it:
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       1. Experiment setup and the HParams experiment summary                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------
Experiment with three hyperparameters in the model:

1. Number of units in the first dense layer
2. Dropout rate in the dropout layer
3. Optimizer

List the values to try, and log an experiment configuration to TensorBoard. 

This step is optional: 
you can provide domain information to enable more precise filtering of hyperparameters in the UI, 
and you can specify which metrics should be displayed.
----------------------------------------------------------------------------------------------------
'''
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

pathHparam = os.path.join(PROJECT_ROOT_DIR, "logs", "hparam_tuning")

with tf.summary.create_file_writer(pathHparam).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

'''
----------------------------------------------------------------------------------------------------
If you choose to skip this step, 
you can use a string literal wherever you would otherwise use an HParam value: e.g., 
hparams['dropout'] instead of hparams[HP_DROPOUT].
----------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       2. Adapt TensorFlow runs to log hyperparameters and metrics                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------
The model will be quite simple: 
two dense layers with a dropout layer between them. 
The training code will look familiar, although the hyperparameters are no longer hardcoded. Instead, 
the hyperparameters are provided in an hparams dictionary and used throughout the training function:
----------------------------------------------------------------------------------------------------
'''
def train_test_model(hparams):
    model = tf.keras.models.Sequential  ([
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
                                        ])
    
    model.compile   (
                    optimizer=hparams[HP_OPTIMIZER],
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    )

    model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy

# For each run, log an hparams summary with the hyperparameters and final accuracy:
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

'''
----------------------------------------------------------------------------------------------------
When training Keras models, you can use callbacks instead of writing these directly:

model.fit(
    ...,
    callbacks=[
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
    ],
)
------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       3. Start runs and log them all under one parent directory                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
------------------------------------------------------------------------------------------------------
You can now try multiple experiments, training each one with a different set of hyperparameters.

For simplicity, use a grid search: 
try all combinations of the discrete parameters and just the lower and upper bounds of the real-valued parameter. 
For more complex scenarios, 
it might be more effective to choose each hyperparameter value randomly (this is called a random search). 
There are more advanced methods that can be used.

Run a few experiments, which will take a few minutes:
------------------------------------------------------------------------------------------------------
'''
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(os.path.join(pathHparam, run_name), hparams)
            session_num += 1

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '      4. Visualize the results in TensorBoard of HParams plugin                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# The HParams dashboard can now be opened. Start TensorBoard and click on "HParams" at the top.

'''
---------------------------------------------------------------------------------------------------------------
The left pane of the dashboard provides filtering capabilities that are active across all the views in the HParams dashboard:

* Filter which hyperparameters/metrics are shown in the dashboard
* Filter which hyperparameter/metrics values are shown in the dashboard
* Filter on run status (running, success, ...)
* Sort by hyperparameter/metric in the table view
* Number of session groups to show (useful for performance when there are many experiments)

The HParams dashboard has three different views, with various useful information:

* The Table View lists the runs, their hyperparameters, and their metrics.
* The Parallel Coordinates View shows each run as a line going through an axis for each hyperparemeter and metric. 
    Click and drag the mouse on any axis to mark a region which will highlight only the runs that pass through it. 
    This can be useful for identifying which groups of hyperparameters are most important. The axes themselves can be re-ordered by dragging them.
* The Scatter Plot View shows plots comparing each hyperparameter/metric with each metric. 
    This can help identify correlations. Click and drag to select a region in a specific plot and highlight those sessions across the other plots.

A table row, a parallel coordinates line, 
and a scatter plot market can be clicked to see a plot of the metrics as a function of training steps for that session 
(although in this tutorial only one step is used for each run).

To further explore the capabilities of the HParams dashboard, download a set of pregenerated logs with more experiments:
-------------------------------------------------------------------------------------------------------------------
'''