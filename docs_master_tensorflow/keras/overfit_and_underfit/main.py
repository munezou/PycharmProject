'''
------------------------------------------------------------------------------------------------------------------------
Know about over-learning and lack of learning.
------------------------------------------------------------------------------------------------------------------------
'''
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

'''
------------------------------------------------------------------------------------------------------------------------
Download IMDB Data Set
------------------------------------------------------------------------------------------------------------------------
'''
import urllib
proxy_support = urllib.request.ProxyHandler({'https': 'http://proxy.kanto.sony.co.jp:10080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

# An error occurs in the original code, so it is a corrected part.(start)
from functools import partial
import numpy as np
np.load = partial(np.load, allow_pickle=True)
# end of modify

imdb = keras.datasets.imdb

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

'''
------------------------------------------------------------------------------------------------------------------------
Survay train_data.
------------------------------------------------------------------------------------------------------------------------
'''
print('---< Output the first 5 train_data. >---')
for i in range(5):
    print ('train_data[{0}] = \n{1}'.format(i, train_data[i]))
print()
print('---< Output the last 5 train_data. >---')
rg = len(train_data)
for i in range(rg -5, rg):
    print('train_data[{0}] = \n{1}'.format(i, train_data[i]))
print()

print('---< Convert an integer to a word and validate the document. >---')
# Dictionary mapping words to integers
word_index = imdb.get_word_index()

# The first one in the index is reserved.
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
word_index = {k:(v+3) for k,v in word_index.items()}

# Swap dictionary keys and values.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print('---< Output the last 5 train_data.(after mapping) >---')
for i in range(5):
    print ('decode_review(train_data[{0}]) = \n{1}'.format(i, decode_review(train_data[i])))
print()
print('---< Output the last 5 train_data.(after mapping) >---')
for i in range(rg -5, rg):
    print('decode_review(train_data[{0}]) = \n{1}'.format(i, decode_review(train_data[i])))
print()

'''
------------------------------------------------------------------------------------------------------------------------
One-Hot en-cording
------------------------------------------------------------------------------------------------------------------------
'''
def multi_hot_sequences(sequences, dimension):
    # Create a matrix of all zeros with shape (len (sequences), dimension).
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        # Set results [i] to 1 for a particular index.
        results[i, word_indices] = 1.0
        #print('result[{0}, {1}] = \n{2}'.format(i, word_indices, results[i, word_indices]))

    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.show()
print()

'''
------------------------------------------------------------------------------------------------------------------------
Over-learning demo

The simplest way to prevent overlearning is to reduce the size of the model, that is, the number of learnable parameters in the model. 
(The number of learning parameters is determined by the number of layers and the number of units per layer.)
In deep learning, the number of learnable parameters of a model is often referred to as the model's “capacity”. 
Intuitively, models with a larger number of parameters have a larger “memory capacity”, making it easier to learn a dictionary-like mapping between training samples and their objective variables. 
This mapping has no generalization ability and is not useful for making predictions with data you have never seen before.

It is important to remember that deep learning models are easy to adapt to training data, but the real challenge is generalization and not adaptation.

On the other hand, if the storage capacity of the network is limited, it is not easy to learn the mapping as described above. 
To reduce the loss, you must learn a compressed representation that is more predictive. 
At the same time, making the model too small makes it difficult to adapt to the training data. 
There is just the right capacity between “too much capacity” and “insufficient capacity”.
------------------------------------------------------------------------------------------------------------------------
'''
# Create a comparison standard.
baseline_model = keras.Sequential([
    # `.summary` を見るために`input_shape`が必要
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

# Start learning
baseline_history = baseline_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
print()

'''
------------------------------------------------------------------------------------------------------------------------
Building smaller models
Let's make a model with a smaller number of hidden units than the reference model just created.
------------------------------------------------------------------------------------------------------------------------
'''
# Create a comparison standard.
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

# Start learning
smaller_history = smaller_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
print()

'''
------------------------------------------------------------------------------------------------------------------------
Building a larger model
As an exercise, you can create a larger model and see how quickly overlearning occurs.
------------------------------------------------------------------------------------------------------------------------
'''
# Create a comparison standard.
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()

# Start learning
bigger_history = bigger_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

print()

'''
------------------------------------------------------------------------------------------------------------------------
Graph loss during training and verification

The solid line is the loss in the training data set and the dashed line is the loss in the validation data set.
(The model with the smaller loss in the verification data is the better model.)
Looking at this, it can be seen that over-learning begins more slowly in the small network than in the comparative model.
(After 6 epochs, not 4 epochs)
Moreover, even after over-learning begins, the decline in performance is slower.
------------------------------------------------------------------------------------------------------------------------
'''
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])

plot_history([('baseline', baseline_history), ('smaller', smaller_history), ('bigger', bigger_history)])

'''
Note that in larger networks, over-learning begins immediately and is strong in one epoch.
The larger the network capacity, the faster the training data will be modeled (resulting in a smaller loss value during training), 
but it will be easier to overlearn (as a result, the loss value during training and the loss during verification)
The value is large and easy to deviate).
'''

'''
------------------------------------------------------------------------------------------------------------------------
strategy

Add weight regularization

Do you know the principle of “Occam's Razor”?
If there are two explanations of something, the most likely explanation is the “simplest” explanation with the least number of assumptions.
This principle also applies to models trained using neural networks.
When there is some training data and network structure, and there are multiple sets of weights that can explain the data (that is, when there are multiple models), 
a simple model is more difficult to overlearn than a complex one .

The “simple model” mentioned here is one that has a small entropy of the distribution of parameter values (or one that has a small number of parameters in the first place as seen above).
Therefore, a general method for alleviating overlearning is to limit only the values with small weights so that the distribution of weight values becomes more orderly (regular).
This is called “weight regularization” and is done by adding the cost associated with the weight to the loss function of the network.
There are two types of costs.

* L1 regularization Adds a cost proportional to the absolute value of the weighting factor.
(This is called “L1 norm” of weight.)

* L2 regularization Adds a cost proportional to the square of the weighting factor.
(We call the square of the weighting factor "L2 norm.")
L2 regularization is called weight decay in neural network terms.
Don't get confused because the name is different. Weight decay is mathematically synonymous with L2 regularization.
------------------------------------------------------------------------------------------------------------------------
'''
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
print()

# Let's look at the impact of L2 regularization.
plot_history([('baseline', baseline_history), ('l2', l2_model_history)])
plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
Add dropout

Dropout is one of the most commonly used neural network regularization techniques.
This technique was developed by Hinton and his students at the University of Toronto.
Dropout is applied to a layer, and it is used to randomly “drop out” (ie zeroize) the features output from the layer during training.
For example, suppose that a layer usually outputs a vector of [0.2, 0.5, 1.3, 0.8, 1.1] for an input sample that is being trained.
When applying dropout, this vector will contain some zeros scattered randomly, eg [0, 0.5, 1.3, 0, 1.1].
"Dropout rate" is the percentage of features that are zeroed out and is usually set between 0.2 and 0.5.
During testing, no units are dropped out, instead the output value is scaled down at the same rate as the dropout rate.
This is to balance the number of active units compared to training.
------------------------------------------------------------------------------------------------------------------------
'''
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

# Start learning
dpt_model_history = dpt_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

plot_history([('baseline', baseline_history), ('dropout', dpt_model_history)])
plt.show()
