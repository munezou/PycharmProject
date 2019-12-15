# tensorflow 2.0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
-----------------------------------------------------------------------------------------------
Better performance with tf.function and AutoGraph

TF 2.0 brings together the ease of eager execution and the power of TF 1.0. 
At the center of this merger is tf.function, which allows you to transform a subset of Python syntax into portable, 
high-performance TensorFlow graphs.

A cool new feature of tf.function is AutoGraph, which lets you write graph code using natural Python syntax. 
For a list of the Python features that you can use with AutoGraph, see AutoGraph Capabilities and Limitations. 
For more details about tf.function, see the RFC TF 2.0: 
Functions, not Sessions. For more details about AutoGraph, see tf.autograph.

This tutorial will walk you through the basic features of tf.function and AutoGraph.
-----------------------------------------------------------------------------------------------
'''

print(__doc__)

# common library
import os
import platform
import shutil
import subprocess
from packaging import version
import timeit

import numpy as np
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The tf.function decorator                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
When you annotate a function with tf.function, you can still call it like any other function. 
But it will be compiled into a graph, which means you get the benefits of faster execution, 
running on GPU or TPU, or exporting to SavedModel.
---------------------------------------------------------------------------------------------------------------
'''
@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer_perform = simple_nn_layer(x, y)
print()
print('simple_nn_layer_perform = \n{0}\n'.format(simple_nn_layer_perform))

'''
----------------------------------------------------------------------------------------------------------------
If we examine the result of the annotation, 
we can see that it's a special callable that handles all interactions with the TensorFlow runtime.
----------------------------------------------------------------------------------------------------------------
'''
print('simple_nn_layer = \n{0}\n'.format(simple_nn_layer))

'''
----------------------------------------------------------------------------------------------------------------
If your code uses multiple functions, 
you don't need to annotate them all - any functions called from an annotated function will also run in graph mode.
----------------------------------------------------------------------------------------------------------------
'''
def linear_layer(x):
    return 2 * x + 1


@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))

deep_net_performance = deep_net(tf.constant((1, 2, 3)))
print('deep_net_perform = {0}\n'.format(deep_net_performance))
print('deep_net = \n{0}\n'.format(deep_net))

'''
---------------------------------------------------------------------------------------------------------------
Functions can be faster than eager code, 
for graphs with many small ops. But for graphs with a few expensive ops (like convolutions), 
you may not see much speedup.
---------------------------------------------------------------------------------------------------------------
'''
conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
    return conv_layer(image)

image = tf.zeros([1, 200, 200, 100])

# warm up
conv_layer(image); conv_fn(image)
print("Eager conv: {0}\n".format(timeit.timeit(lambda: conv_layer(image), number=10)))
print("Function conv: {0}\n".format(timeit.timeit(lambda: conv_fn(image), number=10)))
print("Note how there's not much difference in performance for convolutions")

lstm_cell = tf.keras.layers.LSTMCell(10)

@tf.function
def lstm_fn(input, state):
    return lstm_cell(input, state)

input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2

# warm up
lstm_cell(input, state); lstm_fn(input, state)
print("eager lstm: {0}\n".format(timeit.timeit(lambda: lstm_cell(input, state), number=10)))
print("function lstm: {0}\n".format(timeit.timeit(lambda: lstm_fn(input, state), number=10)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Use Python control flow                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
When using data-dependent control flow inside tf.function, 
you can use Python control flow statements and AutoGraph will convert them into appropriate TensorFlow ops. 
For example, if statements will be converted into tf.cond() if they depend on a Tensor.
---------------------------------------------------------------------------------------------------------------
'''
# In the example below, x is a Tensor but the if statement works as expected:
@tf.function
def square_if_positive(x):
    if x > 0:
        x = x * x
    else:
        x = 0
    return x

print('square_if_positive(2) = {0}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {0}\n'.format(square_if_positive(tf.constant(-2))))

'''
--------------------------------------------------------------------------------------------------------------
Note: 
The previous example uses simple conditionals with scalar values. 
Batching is typically used in real-world code.
--------------------------------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------------------------------
AutoGraph supports common Python statements like while, for, if, break, continue and return, with support for nesting. 
That means you can use Tensor expressions in the condition of while and if statements, or iterate over a Tensor in a for loop.
--------------------------------------------------------------------------------------------------------------
'''
@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

sum_even_tf = sum_even(tf.constant([10, 12, 15, 20]))
print('sum_even_tf = \n{0}'.format(sum_even_tf))
print('sum_even = \n{0}\n'.format(sum_even))

'''
---------------------------------------------------------------------------------------------------------------
AutoGraph also provides a low-level API for advanced users. 
---------------------------------------------------------------------------------------------------------------
'''

# For example we can use it to have a look at the generated code.
print(tf.autograph.to_code(sum_even.python_function))

# Here's an example of more complicated control flow:
@tf.function
def fizzbuzz(n):
    for i in tf.range(n):
        if i % 3 == 0:
            tf.print('Fizz')
        elif i % 5 == 0:
            tf.print('Buzz')
        else:
            tf.print(i)

fizzbuzz_tf_const = fizzbuzz(tf.constant(15))
print('fizzbuzz_tf_const = \n{0}\n'.format(fizzbuzz_tf_const))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Keras and AutoGraph                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
AutoGraph is available by default in non-dynamic Keras models. For more information, see tf.keras.
---------------------------------------------------------------------------------------------------------------
'''

class CustomModel(tf.keras.models.Model):
    
    @tf.function
    def call(self, input_data):
        if tf.math.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data // 2

# create instance
model = CustomModel()

model_tf_const = model(tf.constant([-2, -4]))
print('model_tf_const = {0}\n'.format(model_tf_const))
print('model = \n{0}\n'.format(model))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Side effects                                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Just like in eager mode, 
you can use operations with side effects, like tf.assign or tf.print normally inside tf.function, 
and it will insert the necessary control dependencies to ensure they execute in order.
--------------------------------------------------------------------------------------------------------------
'''
v = tf.Variable(5)

@tf.function
def find_next_odd():
    v.assign(v + 1)
    if v % 2 == 0:
        v.assign(v + 1)


find_next_odd()

print('v = {0}\n'.format(v))
print(tf.autograph.to_code(find_next_odd.python_function))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Debugging                                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------------------------------
tf.function and AutoGraph work by generating code and tracing it into TensorFlow graphs. 
This mechanism does not yet support step-by-step debuggers like pdb. 
However, you can call tf.config.run_functions_eagerly(True) 
to temporarily enable eager execution inside the `tf.function' and use your favorite debugger:
------------------------------------------------------------------------------------------------------------------
'''
@tf.function
def f(x):
    if x > 0:
        # Try setting a breakpoint here!
        # Example:
        #   import pdb
        #   pdb.set_trace()
        x = x + 1
    return x

tf.config.experimental_run_functions_eagerly(True)

# You can now set breakpoints and run the code in a debugger.
f_tf_const = f(tf.constant(1))
print('f_tf_const = {0}\n'.format(f_tf_const))

tf.config.experimental_run_functions_eagerly(False)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Advanced example: An in-graph training loop                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The previous section showed that AutoGraph can be used inside Keras layers and models. 
Keras models can also be used in AutoGraph code.

This example shows how to train a simple Keras model on MNIST with the entire training process?loading batches, 
calculating gradients, updating parameters, calculating validation accuracy, 
and repeating until convergence?is performed in-graph.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download data                                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

train_dataset = mnist_dataset()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the model                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

model = tf.keras.Sequential ((
                            tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
                            tf.keras.layers.Dense(100, activation='relu'),
                            tf.keras.layers.Dense(100, activation='relu'),
                            tf.keras.layers.Dense(10)
                            ))

model.build()

optimizer = tf.keras.optimizers.Adam()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the training loop                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss


@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if step % 10 == 0:
            tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
    return step, loss, accuracy

step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Batching                                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
In real applications batching is essential for performance. 
The best code to convert to AutoGraph is code where the control flow is decided at the batch level. 
If making decisions at the individual example level, try to use batch APIs to maintain performance.
---------------------------------------------------------------------------------------------------------------
'''
# For example, if you have the following code in Python:
def square_if_positive(x):
    return [i ** 2 if i > 0 else i for i in x]

square_if_positive_result = square_if_positive(range(-5, 5))
print('square_if_positive_result = \n{0}\n'.format(square_if_positive_result))

# You may be tempted to write it in TensorFlow as such (and this would work!):
@tf.function
def square_if_positive_naive(x):
    result = tf.TensorArray(tf.int32, size=x.shape[0])
    for i in tf.range(x.shape[0]):
        if x[i] > 0:
            result = result.write(i, x[i] ** 2)
        else:
            result = result.write(i, x[i])
    return result.stack()

square_if_positive_native_result = square_if_positive_naive(tf.range(-5, 5))
print('square_if_positive_native_result = \n{0}\n'.format(square_if_positive_native_result))

# But in this case, it turns out you can write the following:
def square_if_positive_vectorized(x):
    return tf.where(x > 0, x ** 2, x)

square_if_positive_vectorized_result = square_if_positive_vectorized(tf.range(-5, 5))
print('square_if_positive_vectorized_result = \n{0}\n'.format(square_if_positive_vectorized_result))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Re-tracing                                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Key points:

Exercise caution when calling functions with non-tensor arguments, or with arguments that change shapes.
Decorate module-level functions, and methods of module-level classes, and avoid decorating local functions or methods.
tf.function can give you significant speedup over eager execution, at the cost of a slower first-time execution. 
This is because when executed for the first time, the function is also traced into a TensorFlow graph. 
Constructing and optimizing a graph is usually much slower compared to actually executing it:
----------------------------------------------------------------------------------------------------------------
'''
@tf.function
def f(x, y):
    return tf.matmul(x, y)

print(
    "First invocation:",
    timeit.timeit(lambda: f(tf.ones((10, 10)), tf.ones((10, 10))), number=1))

print(
    "Second invocation:",
    timeit.timeit(lambda: f(tf.ones((10, 10)), tf.ones((10, 10))), number=1))

'''
--------------------------------------------------------------------------------------------------------------------
You can easily tell when a function is traced by adding a print statement to the top of the function. 
Because any Python code is only executed at trace time, you will only see the otput of print when the function is traced:
--------------------------------------------------------------------------------------------------------------------
'''
@tf.function
def f():
    print('Tracing!')
    tf.print('Executing')

print('First invocation:')
f()

print('Second invocation:')
f()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       tf.function may also re-trace when called with different non-tensor arguments                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def f(n):
    print(n, 'Tracing!')
    tf.print(n, 'Executing')

f(1)
f(1)

print()

f(2)
f(2)
print()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '        A re-trace can also happen when tensor arguments change shape,                                \n'
        '        unless you specified an input_signature:                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def f(x):
    print(x.shape, 'Tracing!')
    tf.print(x, 'Executing')

f(tf.constant([1]))
f(tf.constant([2]))

print()

f(tf.constant([1, 2]))
f(tf.constant([3, 4]))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '    In addition,                                                                                      \n'
        '    tf.function always creates a new graph function with its own set of traces whenever it is called: \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
def f():
    print('Tracing!')
    tf.print('Executing')

tf.function(f)()
tf.function(f)()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '    This can lead to surprising behavior                                                              \n'
        '    when using the @tf.function decorator in a nested function:                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
def outer():
    @tf.function
    def f():
        print('Tracing!')
        tf.print('Executing')
    f()

outer()
outer()