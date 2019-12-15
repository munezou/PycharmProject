from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

'''
------------------------------------------------------------------------------------------------------------------------
IMDB dataset download
------------------------------------------------------------------------------------------------------------------------
'''
# An error occurs in the original code, so it is a corrected part.(start)
from functools import partial
import numpy as np
np.load = partial(np.load, allow_pickle=True)
# end of modify

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

'''
------------------------------------------------------------------------------------------------------------------------
Examine the data.
------------------------------------------------------------------------------------------------------------------------
'''
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print()
print ('data[0] = \n{0}'.format(train_data[0]))
print ('train_labels[0] = \n{0}'.format(train_labels[0]))
print()

'''
------------------------------------------------------------------------------------------------------------------------
each data of train_data is not same length. (can not calculate model)
------------------------------------------------------------------------------------------------------------------------
'''
print ('len(train_data[0]) = {0},  len(train_data[1]) = {1}'.format(len(train_data[0]), len(train_data[1])))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Convert integers back to words.
------------------------------------------------------------------------------------------------------------------------
'''
# Dictionary mapping words to integers
word_index = imdb.get_word_index()

# The first one in the index is reserved.
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# Replace dictionary index and value.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

'''
------------------------------------------------------------------------------------------------------------------------
A function that converts index to a string and reproduces a sentence using a dataset stored by the dictionary index
------------------------------------------------------------------------------------------------------------------------
'''
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print ('decode_review(train_data[0]) = {0}'.format(decode_review(train_data[0])))

'''
------------------------------------------------------------------------------------------------------------------------
prepare model data.
------------------------------------------------------------------------------------------------------------------------
'''
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# confirm the length of data betwenn train_data and test_data.
print ('len(train_data[0]) = {0},  len(train_data[1]) = {1}'.format(len(train_data[0]), len(train_data[1])))
print()
print('train_data[0] = \n{0}'.format(train_data[0]))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Model construction¶
A neural network consists of stacking layers. This requires two major decisions.

* How many layers do you have in your model?
* How many hidden units are used per layer?
In this example, the input data consists of an array of word indexes. The label to be estimated is 0 or 1. Let's build a model for this problem.
------------------------------------------------------------------------------------------------------------------------
'''
# Input format is the number of vocabulary used in movie review (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
'''
explanation)
------------------------------------------------------------------------------------------------------------------------
These layers are stacked in a row to form a classifier.

1. The first layer is the Embedding layer. 
   This layer takes an integer encoded vocabulary and searches for an embedded vector corresponding to each word index. 
   Embedded vectors are learned in model training. An additional dimension is added to the output matrix for vectorization. 
   As a result, the dimensions are (batch, sequence, embedding).
2. Next is the GlobalAveragePooling1D (one-dimensional global average pooling) tier.
   This layer finds, for each sample, the mean value in the dimensional direction of the sequence and returns a vector of fixed length.
   This results in the model being able to handle variable-length input in its simplest form.
3. This fixed-length output vector is passed to the all coupled (Dense) layer with 16 hidden units.
4. The last layer is fully coupled to one output node. 
   By using the sigmoid activation function, the value is a floating point number between 0 and 1 representing probability or confidence.
------------------------------------------------------------------------------------------------------------------------   
Hidden unit
The model above has two middle or "hidden" layers between the input and output.
The output (unit, node or neuron) is the number of dimensions of the internal representation of the layer.
In other words, this network is the degree of freedom when learning internal expression.

The network may learn more complex internal representations if there are more hidden units in the model (if the dimensionality of the internal representation space is larger) and / or if there are more layers I can do it.
However, as a result, in addition to the computational complexity of the network being increased, it becomes possible to learn patterns that you do not want to learn.
Patterns that you do not want to learn are patterns that improve the performance of training data but do not improve the performance of test data.
This problem is called overfitting. This issue will be examined later.
------------------------------------------------------------------------------------------------------------------------
Loss Function and Optimizer¶
To train the model, you need a loss function and an optimizer.
Since this problem is a binary classification problem, and the output of the model is probability (one unit of layer and sigmoid activation function), 
we will use the binary_crossentropy (binary cross entropy) function as the loss function.

This is not the only loss function candidate.
For example, you can use mean_squared_error (mean squared error).
However, in general, binary_crossentropy is better for dealing with probabilities.
binary_crossentropy is a measure that measures the "distance" between probability distributions.
In this case, it is the distance between the true distribution and the distribution of predicted values.

Later on, when examining regression problems (for example, estimating house prices), you will see the use of another loss function, mean_squared_error.

Now let's set up the model's optimizer and loss function.
------------------------------------------------------------------------------------------------------------------------
'''
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

'''
------------------------------------------------------------------------------------------------------------------------
Create data for verification
When training, I would like to verify the accuracy rate on data that the model does not see. 
Separate 10,000 samples from the original training data to create a validation set. 
(Why not use test data here? The purpose of this time is to develop and tune the model using only training data,
 and then use the test data only once and the accuracy rate To verify)
 -----------------------------------------------------------------------------------------------------------------------
'''
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

'''
------------------------------------------------------------------------------------------------------------------------
Model training
Train a 40 epoch model using a mini-batch of 512 samples. 
This results in 40 iterations of all the samples in x_train and y_train. 
During training, we will monitor model loss and accuracy rates using 10,000 samples of validation data.
------------------------------------------------------------------------------------------------------------------------
'''
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

'''
------------------------------------------------------------------------------------------------------------------------
Model evaluation
Well, let's look at the performance of the model. Two values are returned. 
It is a loss (it is a numerical value which shows an error, and a smaller one is better) and an accuracy rate.
------------------------------------------------------------------------------------------------------------------------
'''
results = model.evaluate(test_data, test_labels)

print('result[error, accuracy] = {0}'.format(results))

'''
------------------------------------------------------------------------------------------------------------------------
Draw a time series graph of accuracy rate and loss¶
model.fit () returns a History object containing a dictionary that records everything that occurred during training.
------------------------------------------------------------------------------------------------------------------------
'''
history_dict = history.history
print('history_dict.keys() = \n{0}'.format(history_dict.keys()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
There are 4 entries. 
Each indicates the indicator that was being monitored during training and validation. 
You can use this to create graphs that compare training and verification losses and graphs that compare training and verification accuracy rates.
------------------------------------------------------------------------------------------------------------------------
'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" は青いドット
plt.plot(epochs, loss, 'bo', label='Training loss')
# ”b" は青い実線
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear diagram
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()