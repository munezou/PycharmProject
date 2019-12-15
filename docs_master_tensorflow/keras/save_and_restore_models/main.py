'''
------------------------------------------------------------------------------------------------------------------------
Get sample dataset.
------------------------------------------------------------------------------------------------------------------------
'''
import os

import tensorflow as tf
from tensorflow import keras

# Import helper library
import numpy as np
import matplotlib.pyplot as plt

print ('---< Tensorflow condition >---')
print('tf.__version__ = {0}'.format(tf.__version__))
print('keras.__version__ = {0}'.format(keras.__version__))
print()

# setting proxy
import urllib
proxy_support = urllib.request.ProxyHandler({'https': 'http://proxy.kanto.sony.co.jp:10080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)
# end of setting proxy

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# to confirm image file.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

train_images_org = train_images / 255.0
test_images_org = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images_org[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

'''
------------------------------------------------------------------------------------------------------------------------
Model definition
------------------------------------------------------------------------------------------------------------------------
'''
# A function that returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.sparse_categorical_crossentropy,
                    metrics=['accuracy'])

    return model


# Create a basic model instance.
model = create_model()
model.summary()

'''
------------------------------------------------------------------------------------------------------------------------
Save checkpoints during training.

The main use is to automatically save checkpoints during or after training.
This allows you to use the model without retraining, and if training is interrupted, you can resume where you left off.

tf.keras.callbacks.ModelCheckpoint is the callback to do this.
This callback has several arguments for constructing a checkpoint.
------------------------------------------------------------------------------------------------------------------------
'''
# How to use CheckPointCall.
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Make a checkpoint callback.
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model = create_model()

# Give the training a callback.
model.fit(train_images, train_labels,  epochs = 10, validation_data = (test_images,test_labels), callbacks = [cp_callback])

# You might see a warning about saving the optimizer state.
# These warnings (including similar warnings that occur in this notebook) are intended to deprecate old usage and can be ignored.

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Checkpoint callback options

This callback has options to give the checkpoint a unique name and adjust the frequency of the checkpoint.
------------------------------------------------------------------------------------------------------------------------
'''
# Embed the epoch number (using `str.format`) in the file name.
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weight every 5 epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels, epochs = 50, callbacks = [cp_callback], validation_data = (test_images,test_labels), verbose=0)

# Next, check the completed checkpoint and select the last one.
latest = tf.train.latest_checkpoint(checkpoint_dir)
print('latest = {0}'.format(latest))
print()

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Save weights manually.
------------------------------------------------------------------------------------------------------------------------
'''
print('---< save weight manually. >---')
# Save weight
model.save_weights('./checkpoints/my_checkpoint')

# Weight restoration
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Saving the entire model

You can save the entire model to a file, including weight values, model settings, and even optimizer settings.
This allows you to save the state of the model at a certain point and resume training where you left off without having to access the original Python code.

(If you are saving an HDF5 file to a model that uses an optimizer that is not an optimizer included in the tf.train module, you can save the optimizer settings.)

It is convenient to be able to save a fully functional model.
The saved model can be loaded with TensorFlow.js (HDF5, Saved Model), trained and executed in the browser, 
and can be executed on mobile devices using TensorFlow Lite (HDF5, Saved Model) It can also be converted.
------------------------------------------------------------------------------------------------------------------------
'''
# Keras supports basic file formats using the HDF5 standard.
# For our purposes, the saved model can be treated as a single binary large object (blob).
model = create_model()

model.fit(train_images, train_labels, epochs=50)

# Save the entire model in one HDF5 file.
model.save('my_model.h5')

'''
------------------------------------------------------------------------------------------------------------------------
Recreate the model using the saved file.
------------------------------------------------------------------------------------------------------------------------
'''
# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# Check the accuracy rate.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print()

# Let them try it out.
predictions = new_model.predict(test_images)

print('predictions[0] = \n{0}'.format(predictions[0]))

print('np.argmax(predictions[0]) = {0}'.format(np.argmax(predictions[0])))

print('test_labels[0] = {0}'.format(test_labels[0]))

print()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images_org)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
another method : saved_model
------------------------------------------------------------------------------------------------------------------------
'''
model = create_model()

model.fit(train_images, train_labels, epochs=50)

# create saved model
saved_model_path = tf.keras.experimental.export_saved_model(model, "./saved_models")

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
print('new_model = \n{0}'.format(new_model))
print()

# Run the restored model.

# You must compile before evaluating the model.
# This step is not necessary if you are only deploying the model.

new_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# Evaluate the model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))