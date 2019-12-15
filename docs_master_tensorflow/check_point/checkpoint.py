'''
---------------------------------------------------------------------------------------------------------------------
Training checkpoints 

over view)
The phrase "Saving a TensorFlow model" typically means one of two things:

1. Checkpoints, OR
2. SavedModel.

Checkpoints capture the exact value of all parameters (tf.Variable objects) used by a model. 
Checkpoints do not contain any description of the computation defined by the model and thus are typically only useful when source code 
that will use the saved parameter values is available.

The SavedModel format on the other hand includes a serialized description of the computation defined by the model 
in addition to the parameter values (checkpoint). 
Models in this format are independent of the source code that created the model. 
They are thus suitable for deployment via TensorFlow Serving, TensorFlow Lite, TensorFlow.js, 
or programs in other programming languages (the C, C++, Java, Go, Rust, C# etc. TensorFlow APIs).

This guide covers APIs for writing and reading checkpoints.
------------------------------------------------------------------------------------------------------------------------
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import os
import platform
import shutil
import subprocess
from packaging import version
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np


import tensorflow as tf
import tensorflow.compat.v1 as tf_compat

print(__doc__)

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

class Net(tf.keras.Model):
    """A simple linear model."""

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)

net = Net()
print('net = \n{0}\n'.format(net))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Saving from tf.keras training APIs                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# tf.keras.Model.save_weights saves a TensorFlow checkpoint.
easy_check_path = os.path.join(PROJECT_ROOT_DIR, 'ckpt', 'easy_checkpoint')
net.save_weights(easy_check_path)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Writing checkpoints                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The persistent state of a TensorFlow model is stored in tf.Variable objects. 
These can be constructed directly, but are often created through high-level APIs like tf.keras.layers or tf.keras.Model.

The easiest way to manage variables is by attaching them to Python objects, then referencing those objects.

Subclasses of tf.train.Checkpoint, tf.keras.layers.Layer, and tf.keras.Model automatically track variables assigned to their attributes. 
The following example constructs a simple linear model, then writes checkpoints which contain values for all of the model's variables.

You can easily save a model-checkpoint with Model.save_weights
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Manual checkpointing                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# To help demonstrate all the features of tf.train.Checkpoint define a toy dataset and optimization step:
def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(dict(x=inputs, y=labels)).repeat(10).batch(2)

def train_step(net, example, optimizer):
    """Trains `net` on `example` using `optimizer`."""
    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))
        
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create the checkpoint objects                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To manually make a checkpoint you will need a tf.train.Checkpoint object. 
Where the objects you want to checkpoint are set as attributes on the object.

A tf.train.CheckpointManager can also be helpful for managing multiple checkpoints.
---------------------------------------------------------------------------------------------------------------
'''
opt = tf.keras.optimizers.Adam(0.1)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
tf_check_path = os.path.join(PROJECT_ROOT_DIR, "tf_ckpt")
manager = tf.train.CheckpointManager(ckpt, tf_check_path, max_to_keep=3)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train and checkpoint the model                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The following training loop creates an instance of the model and of an optimizer, then gathers them into a tf.train.Checkpoint object. 
It calls the training step in a loop on each batch of data, and periodically writes checkpoints to disk.
---------------------------------------------------------------------------------------------------------------
'''
def train_and_checkpoint(net, manager):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for example in toy_dataset():
        loss = train_step(net, example, opt)
        ckpt.step.assign_add(1)
        
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(loss.numpy()))

train_and_checkpoint(net, manager)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Restore and continue training                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# After the first you can pass a new model and manager, but pickup training exactly where you left off:
opt = tf.keras.optimizers.Adam(0.1)
net = Net()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, tf_check_path, max_to_keep=3)

train_and_checkpoint(net, manager)

'''
---------------------------------------------------------------------------------------------------------------
The `tf.train.CheckpointManager` object deletes old checkpoints. 
Above it's configured to keep only the three most recent checkpoints.
---------------------------------------------------------------------------------------------------------------
'''
print(manager.checkpoints)  # List the three remaining checkpoints

'''
---------------------------------------------------------------------------------------------------------------
These paths, e.g. './tf_ckpts/ckpt-10', are not files on disk. 
Instead they are prefixes for an index file and one or more data files which contain the variable values. 
These prefixes are grouped together in a single checkpoint file ('./tf_ckpts/checkpoint') where the CheckpointManager saves its state.
---------------------------------------------------------------------------------------------------------------
'''

'''
----------------------------------------------------------------------------------------------------------------
Loading mechanics

TensorFlow matches variables to checkpointed values by traversing a directed graph with named edges, 
starting from the object being loaded. Edge names typically come from attribute names in objects, 
for example the "l1" in self.l1 = tf.keras.layers.Dense(5). tf.train.Checkpoint uses its keyword argument names, as in the "step" in tf.train.Checkpoint(step=...).

The dependency graph from the example above looks like this:
'''
im = Image.open(os.path.join(PROJECT_ROOT_DIR, "image", "whole_checkpoint.png"))
im.show()
'''
Visualization of the dependency graph for the example training loop

With the optimizer in red, regular variables in blue, and optimizer slot variables in orange. 
The other nodes, for example representing the tf.train.Checkpoint, are black.

Slot variables are part of the optimizer's state, but are created for a specific variable. 
For example the 'm' edges above correspond to momentum, which the Adam optimizer tracks for each variable. 
-----------------------------------------------------------------------------------------------------------------
'''

'''
-----------------------------------------------------------------------------------------------------------------
Calling restore() on a tf.train.Checkpoint object queues the requested restorations, 
restoring variable values as soon as there's a matching path from the Checkpoint object. 
For example we can load just the kernel from the model we defined above by reconstructing one path to it through the network and the layer.
----------------------------------------------------------------------------------------------------------------
'''
to_restore = tf.Variable(tf.zeros([5]))
print('to_restore.numpy() = \n{0}\n'.format(to_restore.numpy()))  # All zeros
fake_layer = tf.train.Checkpoint(bias=to_restore)
fake_net = tf.train.Checkpoint(l1=fake_layer)
new_root = tf.train.Checkpoint(net=fake_net)
status = new_root.restore(tf.train.latest_checkpoint(tf_check_path))
print('to_restore.numpy() = \n{0}\n'.format(to_restore.numpy()))  # We get the restored value now

'''
----------------------------------------------------------------------------------------------------------------
The dependency graph for these new objects is a much smaller subgraph of the larger checkpoint we wrote above. 
It includes only the bias and a save counter that tf.train.Checkpoint uses to number checkpoints.
'''
im = Image.open(os.path.join(PROJECT_ROOT_DIR, "image", "partial_checkpoint.png"))
im.show()
'''
restore() returns a status object, which has optional assertions. 
All of the objects we've created in our new Checkpoint have been restored, 
so status.assert_existing_objects_matched() passes.
----------------------------------------------------------------------------------------------------------------
'''
print('status.assert_existing_objects_matched() = \n{0}\n'.format(status.assert_existing_objects_matched()))

'''
----------------------------------------------------------------------------------------------------------------
There are many objects in the checkpoint which haven't matched, 
including the layer's kernel and the optimizer's variables. 
status.assert_consumed() only passes if the checkpoint and the program match exactly, and would throw an exception here.
----------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Delayed restorations                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Layer objects in TensorFlow may delay the creation of variables to their first call, when input shapes are available. 
For example the shape of a Dense layer's kernel depends on both the layer's input and output shapes, 
and so the output shape required as a constructor argument is not enough information to create the variable on its own. 
Since calling a Layer also reads the variable's value, a restore must happen between the variable's creation and its first use.

To support this idiom, tf.train.Checkpoint queues restores which don't yet have a matching variable.
---------------------------------------------------------------------------------------------------------------
'''
delayed_restore = tf.Variable(tf.zeros([1, 5]))
print('delayed_restore.numpy() = {0}\n'.format(delayed_restore.numpy()))  # Not restored; still zeros
fake_layer.kernel = delayed_restore
print('delayed_restore.numpy() = {0}\n'.format(delayed_restore.numpy()))   # Restored

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Manually inspecting checkpoints                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
tf.train.list_variables lists the checkpoint keys and shapes of variables in a checkpoint. 
Checkpoint keys are paths in the graph displayed above.
---------------------------------------------------------------------------------------------------------------
'''
tf_train_list_variables = tf.train.list_variables(tf.train.latest_checkpoint(tf_check_path))
print('tf_train_list_variables = \n{0}\n'.format(tf_train_list_variables))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       List and dictionary tracking                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
As with direct attribute assignments like self.l1 = tf.keras.layers.Dense(5), 
assigning lists and dictionaries to attributes will track their contents.
---------------------------------------------------------------------------------------------------------------
'''
save = tf.train.Checkpoint()
save.listed = [tf.Variable(1.)]
save.listed.append(tf.Variable(2.))
save.mapped = {'one': save.listed[0]}
save.mapped['two'] = save.listed[1]
save_path = save.save(os.path.join(PROJECT_ROOT_DIR, "tf_list", "tf_list_example"))

restore = tf.train.Checkpoint()
v2 = tf.Variable(0.)
assert 0. == v2.numpy()  # Not restored yet
restore.mapped = {'two': v2}
restore.restore(save_path)
assert 2. == v2.numpy()

'''
---------------------------------------------------------------------------------------------------------------
You may notice wrapper objects for lists and dictionaries.
These wrappers are checkpointable versions of the underlying data-structures. 
Just like the attribute based loading, these wrappers restore a variable's value as soon as it's added to the container.
---------------------------------------------------------------------------------------------------------------
'''
restore.listed = []
print('restore.listed = {0}\n'.format(restore.listed))  # ListWrapper([])
v1 = tf.Variable(0.)
restore.listed.append(v1)  # Restores v1, from restore() in the previous cell
assert 1. == v1.numpy()

'''
---------------------------------------------------------------------------------------------------------------
The same tracking is automatically applied to subclasses of tf.keras.Model, 
and may be used for example to track lists of layers.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Saving object-based checkpoints with Estimator                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Estimators by default save checkpoints with variable names rather than the object graph described in the previous sections. 
tf.train.Checkpoint will accept name-based checkpoints, 
but variable names may change when moving parts of a model outside of the Estimator's model_fn. 
Saving object-based checkpoints makes it easier to train a model inside an Estimator and then use it outside of one.
--------------------------------------------------------------------------------------------------------------
'''

def model_fn(features, labels, mode):
    net = Net()
    opt = tf.keras.optimizers.Adam(0.1)
    ckpt = tf.train.Checkpoint(step=tf_compat.train.get_global_step(),optimizer=opt, net=net)
    
    with tf.GradientTape() as tape:
        output = net(features['x'])
        loss = tf.reduce_mean(tf.abs(output - features['y']))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    
    return tf.estimator.EstimatorSpec(
        mode, 
        loss=loss, 
        train_op=tf.group(opt.apply_gradients(zip(gradients, variables)), 
        ckpt.step.assign_add(1)),
        # Tell the Estimator to save "ckpt" in an object-based format.
        scaffold=tf_compat.train.Scaffold(saver=ckpt)
        )

tf.keras.backend.clear_session()
tf_estimsate_path = os.path.join(PROJECT_ROOT_DIR, "tf_estimator_example", "")
est = tf.estimator.Estimator(model_fn, tf_estimsate_path)
est_train = est.train(toy_dataset, steps=10)
print('est_train = \n{0}\n'.format(est_train))

# tf.train.Checkpoint can then load the Estimator's checkpoints from its model_dir.
opt = tf.keras.optimizers.Adam(0.1)
net = Net()
ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt, net=net)
ckpt.restore(tf.train.latest_checkpoint(tf_estimsate_path))
ckpt_step_numpy = ckpt.step.numpy()  # From est.train(..., steps=10)
print('ckpt_step_numpy = {0}\n'.format(ckpt_step_numpy))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Summary                                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
TensorFlow objects provide an easy automatic mechanism for saving and restoring the values of variables they use.
---------------------------------------------------------------------------------------------------------------
'''