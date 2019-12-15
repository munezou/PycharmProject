# tensorflow 2.0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
-----------------------------------------------------------------------------------------------
base operation of Eager_Execution
-----------------------------------------------------------------------------------------------
'''
print(__doc__)

# common library
import time
import os
import platform
import shutil
import subprocess
from packaging import version

import matplotlib.pyplot as plt

import numpy as np


import tensorflow as tf
import cProfile

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Start eager execution                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {0}".format(m))

a = tf.constant([[1, 2], [3, 4]])
print('a = \n{0}\n'.format(a))

# suport borad cast
b = tf.add(a, 1)
print('b = \n{0}\n'.format(b))

# Operator overloading is supported.
print('a * b = \n{0}\n'.format(a * b))

c = np.matmul(a, b)
print('c = \n{0}\n'.format(c))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dynamic control flow                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The main advantage of Eager Execution is that all the functionality of the host language is available when executing the model. 
For example, you can easily write fizzbuzz:
---------------------------------------------------------------------------------------------------------------
'''
def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('FizzBuzz')
        elif int(num % 3) == 0:
            print('Fizz')
        elif int(num % 5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        
        counter += 1

print('fizzbuzz(15) = {0}\n'.format(fizzbuzz(15)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Eager training                                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Computing gradients                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Automatic differentiation is useful for implementing machine learning algorithms such as backpropagation for training neural networks. 
During eager execution, use tf.GradientTape to trace operations for computing gradients later.

You can use tf.GradientTape to train and/or compute gradients in eager. 
It is especially useful for complicated training loops.

Since different operations can occur during each call, all forward-pass operations get recorded to a "tape". 
To compute the gradient, play the tape backwards and then discard. 
A particular tf.GradientTape can only compute one gradient; subsequent calls throw a runtime error.
---------------------------------------------------------------------------------------------------------------
'''
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print('grad = {0}\n'.format(grad))  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train a model                                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------------------------------
The following example creates a multi-layer model that classifies the standard MNIST handwritten digits. 
It demonstrates the optimizer and layer APIs to build trainable graphs in an eager execution environment.
-----------------------------------------------------------------------------------------------------------------
'''
# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices((
                                            tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
                                            tf.cast(mnist_labels,tf.int64)
                                            ))

dataset = dataset.shuffle(1000).batch(32)

# Build the model
mnist_model = tf.keras.Sequential   ([
                                    tf.keras.layers.Conv2D(16,[3,3], activation='relu', input_shape=(None, None, 1)),
                                    tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
                                    tf.keras.layers.GlobalAveragePooling2D(),
                                    tf.keras.layers.Dense(10)
                                    ])

'''
--------------------------------------------------------------------------------------------------------------------
Even without training, call the model and inspect the output in eager execution:
--------------------------------------------------------------------------------------------------------------------
'''
for images,labels in dataset.take(1):
    print("Logits: \n{0}\n".format(mnist_model(images[0:1]).numpy()))

'''
--------------------------------------------------------------------------------------------------------------------
While keras models have a builtin training loop (using the fit method), 
sometimes you need more customization. 

Here's an example, of a training loop implemented with eager:
--------------------------------------------------------------------------------------------------------------------
'''
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []

'''
--------------------------------------------------------------------------------------------------------------------
Note: 
Use the assert functions in tf.debugging to check if a condition holds up. 
This works in eager and graph execution.
--------------------------------------------------------------------------------------------------------------------
'''
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
    
        # Add asserts to check the shape of the output.
        tf.debugging.assert_equal(logits.shape, (32, 10))
        
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

def train(epochs):
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels)
            
        print ('Epoch {} finished'.format(epoch))

train(epochs = 3)

plt.figure(figsize=(8, 6))
plt.title("Relationship diagram between bach and loss")
plt.plot(loss_history)
plt.xlim([0, 6000])
plt.ylim([0, 2.4])
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.grid(True)
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Variables and optimizers                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
tf.Variable objects store mutable tf.Tensor-like values accessed during training to make automatic differentiation easier.

The collections of variables can be encapsulated into layers or models, along with methods that operate on them. 
See Custom Keras layers and models for details. 
The main difference between layers and models is that models add methods like Model.fit, Model.evaluate, and Model.save.
----------------------------------------------------------------------------------------------------------------
'''
# For example, the automatic differentiation example above can be rewritten:
class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(10., name='bias')
        
    def call(self, inputs):
        return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# disply raw datas(3 * x + 2)
plt.figure(figsize=(8, 6))
plt.title("raw datas for 3 * x + 2")
plt.scatter(training_inputs, training_outputs, c = 'blue')
plt.grid(True)
plt.xlabel("training_inputs")
plt.ylabel("training_outputs")
plt.show()

# The loss function to be optimized
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])

'''
--------------------------------------------------------------------------------------------------------------------
Next:

1. Create the model.
2. The Derivatives of a loss function with respect to model parameters.
3. A strategy for updating the variables based on the derivatives.
--------------------------------------------------------------------------------------------------------------------
'''
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

steps = 300
for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

'''
--------------------------------------------------------------------------------------------------------------------
Note: 
Variables persist until the last reference to the python object is removed, and is the variable is deleted.
--------------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Object-based saving                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

# A tf.keras.Model includes a covienient save_weights method allowing you to easily create a checkpoint:
model.save_weights('weights')
status = model.load_weights('weights')

'''
------------------------------------------------------------------------------------------------------------------
Using tf.train.Checkpoint you can take full control over this process.
------------------------------------------------------------------------------------------------------------------
'''
# This section is an abbreviated version of the guide to training checkpoints.
x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)

x.assign(2.)   # Assign a new value to the variables and save.
checkpoint_path = os.path.join(PROJECT_ROOT_DIR, "ckpt", "")
checkpoint.save(checkpoint_path)

x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print('x = {0}\n'.format(x))  # => 2.0

'''
-------------------------------------------------------------------------------------------------------------------
To save and load models, tf.train.Checkpoint stores the internal state of objects, without requiring hidden variables. 
To record the state of a model, an optimizer, and a global step, pass them to a tf.train.Checkpoint:
-------------------------------------------------------------------------------------------------------------------
'''
model = tf.keras.Sequential ([
                            tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
                            tf.keras.layers.GlobalAveragePooling2D(),
                            tf.keras.layers.Dense(10)
                            ])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

checkpoint_dir = os.path.join(PROJECT_ROOT_DIR, "path", "to", "model_dir")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt", "")
root = tf.train.Checkpoint(optimizer=optimizer, model=model)

root.save(checkpoint_prefix)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

'''
--------------------------------------------------------------------------------------------------------------------
Note: 
In many training loops, variables are created after tf.train.Checkpoint.restore is called. 
These variables will be restored as soon as they are created, 
and assertions are available to ensure that a checkpoint has been fully loaded.
See the guide to training checkpoints for details.
--------------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Object-oriented metrics                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
-------------------------------------------------------------------------------------------------------------------
tf.keras.metrics are stored as objects. 
Update a metric by passing the new data to the callable, 
and retrieve the result using the tf.keras.metrics.result method, for example:
-------------------------------------------------------------------------------------------------------------------
'''
m = tf.keras.metrics.Mean("loss")
m(0)
m(5)
print('m.result() = {0}\n'.format(m.result())) # => 2.5

m([8, 9])
print('m.result() = {0}\n'.format(m.result()))  # => 5.5

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Summaries and TensorBoard                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
TensorBoard is a visualization tool for understanding, debugging and optimizing the model training process. 
It uses summary events that are written while executing the program.

You can use tf.summary to record summaries of variable in eager execution. 
For example, to record summaries of loss once every 100 training steps:
----------------------------------------------------------------------------------------------------------------
'''
# Delete past tensorboard results.
pathLogs = os.path.join(PROJECT_ROOT_DIR, "tb")
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

logdir = os.path.join(PROJECT_ROOT_DIR, "tb", "")
writer = tf.summary.create_file_writer(logdir)

steps = 1000
with writer.as_default():  # or call writer.set_as_default() before the loop.
    for i in range(steps):
        step = i + 1
        # Calculate loss with your real train function.
        loss = 1 - 0.001 * step
        if step % 100 == 0:
            tf.summary.scalar('loss', loss, step=step)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Advanced automatic differentiation topics                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dynamic models                                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
tf.GradientTape can also be used in dynamic models. 
This example for a backtracking line search algorithm looks like normal NumPy code, 
except there are gradients and is differentiable, despite the complex control flow:
---------------------------------------------------------------------------------------------------------------
'''
def line_search_step(fn, init_x, rate=1.0):
    with tf.GradientTape() as tape:
        # Variables are automatically tracked.
        # But to calculate a gradient from a tensor, you must `watch` it.
        tape.watch(init_x)
        value = fn(init_x)
    grad = tape.gradient(value, init_x)
    grad_norm = tf.reduce_sum(grad * grad)
    init_value = value
    while value > init_value - rate * grad_norm:
        x = init_x - rate * grad
        value = fn(x)
        rate /= 2.0
    return x, value

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Custom gradients                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Custom gradients are an easy way to override gradients. 
Within the forward function, define the gradient with respect to the inputs, outputs, or intermediate results. 
For example, here's an easy way to clip the norm of the gradients in the backward pass:
---------------------------------------------------------------------------------------------------------------
'''
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
    y = tf.identity(x)
    def grad_fn(dresult):
        return [tf.clip_by_norm(dresult, norm), None]
    return y, grad_fn

'''
----------------------------------------------------------------------------------------------------------------
Custom gradients are commonly used to provide a numerically stable gradient for a sequence of operations:
----------------------------------------------------------------------------------------------------------------
'''
def log1pexp(x):
    return tf.math.log(1 + tf.exp(x))

def grad_log1pexp(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = log1pexp(x)
    return tape.gradient(value, x)

# The gradient computation works fine at x = 0.
grad_log1_0 = grad_log1pexp(tf.constant(0.)).numpy()
print('grad_log1_0 = {0}\n'.format(grad_log1_0))

# However, x = 100 fails because of numerical instability.
grad_log1_100 = grad_log1pexp(tf.constant(100.)).numpy()
print('grad_log1_100 = {0}\n'.format(grad_log1_100))

'''
----------------------------------------------------------------------------------------------------------------
Here, the log1pexp function can be analytically simplified with a custom gradient. 
The implementation below reuses the value for tf.exp(x) that is computed 
during the forward pass—making it more efficient by eliminating redundant calculations:
----------------------------------------------------------------------------------------------------------------
'''
@tf.custom_gradient
def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
        return dy * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad

def grad_log1pexp(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = log1pexp(x)
    return tape.gradient(value, x)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Performance                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Computation is automatically offloaded to GPUs during eager execution. 
If you want control over where a computation runs you can enclose it in a tf.device('/gpu:0') block (or the CPU equivalent):
---------------------------------------------------------------------------------------------------------------
'''
def measure(x, steps):
    # TensorFlow initializes a GPU the first time it's used, exclude from timing.
    tf.matmul(x, x)
    start = time.time()
    for i in range(steps):
        x = tf.matmul(x, x)
        
    # tf.matmul can return before completing the matrix multiplication
    # (e.g., can return after enqueing the operation on a CUDA stream).
    # The x.numpy() call below will ensure that all enqueued operations
    # have completed (and will also copy the result to host memory,
    # so we're including a little more than just the matmul operation
    # time).
    
    _ = x.numpy()
    end = time.time()
    return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
    print("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))

# Run on GPU, if available:
if tf.config.experimental.list_physical_devices("GPU"):
    with tf.device("/gpu:0"):
        print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
else:
    print("GPU: not found")

# A tf.Tensor object can be copied to a different device to execute its operations:
if tf.config.experimental.list_physical_devices("GPU"):
    x = tf.random.normal([10, 10])

    x_gpu0 = x.gpu()
    x_cpu = x.cpu()

    _ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
    _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

'''
------------------------------------------------------------------------------------------------------
Benchmarks
For compute-heavy models, such as ResNet50 training on a GPU, eager execution performance is comparable to tf.function execution. 
But this gap grows larger for models with less computation and there is work to be done for optimizing hot code paths for models with lots of small operations.

Work with functions
While eager execution makes development and debugging more interactive, 
TensorFlow 1.x style graph execution has advantages for distributed training, performance optimizations, and production deployment. 
To bridge this gap, TensorFlow 2.0 introduces functions via the tf.function API. For more information, see the tf.function guide.
--------------------------------------------------------------------------------------------------------
'''
