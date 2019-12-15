# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print ('train_images.shape = {0}'.format(train_images.shape))

print ('len(train_labels) = {0}'.format(len(train_labels)))

print ('train_labels = {0}'.format(train_labels))

print()

print ('test_images.shape = {0}'.format(test_images.shape))

print ('len(test_labels) = {0}'.format(len(test_labels)))

print()

# Data preprocessing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
It is important to preprocess the training data set and the test data set in the same way.
------------------------------------------------------------------------------------------------------------------------
'''
train_images = train_images / 255.0

test_images = test_images / 255.0

'''
------------------------------------------------------------------------------------------------------------------------
Let's display the first 25 images of the training data set with class names. 
Make sure your data is in the correct format before building and training your network.
------------------------------------------------------------------------------------------------------------------------
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

'''
-------------------------------------------------------------------------------------------------------------------------
 Create model
 
 The first layer of this network is tf.keras.layers.Flatten. 
 This layer transforms the image from a two-dimensional array (28 x 28 pixels) to a one-dimensional array 28 x 28 = 784 pixels. 
 Think of this layer as dropping the rows of pixels stacked in the image and aligning them side by side. 
 There are no parameters to learn in this layer, just format conversion of the data.

 After the pixels are one-dimensionalized, the network becomes two tf.keras.layers.Dense layers.
 These layers are layers of tightly coupled or fully coupled neurons. 
 The first Dense layer has 128 nodes (or neurons). 
 The second layer, which is also the last layer, is the 10-node softmax layer. 
 This layer returns an array of 10 probabilities that sum to one. 
 Each node outputs the probability that the image you are looking at belongs to one of ten classes.
 ------------------------------------------------------------------------------------------------------------------------
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''
------------------------------------------------------------------------------------------------------------------------
Model compilation¶
You will need to add some settings to be able to train the model. These settings are added at compile time of the model.

*Loss function — Measures how accurate the model is during training. 
                 By minimizing the value of this function, you are aiming the model being trained in the right direction.
*Optimizer     —The data that the model sees and the value of the loss function determine how to update the model.
*Metrics       — used to monitor training and testing steps. 
                In the example below,
                we use accuracy (accuracy rate), that is, the rate at which the image was correctly classified.
------------------------------------------------------------------------------------------------------------------------
'''
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''
------------------------------------------------------------------------------------------------------------------------
Model training
Training a neural network involves the following steps:

1. Populate the model with training data—in this example, two arrays, train_images and train_labels.
2. The model learns the correspondence between images and labels.
3. Let the model predict (classify) the test data set—in this example, the test_images array.
   Then match the test results with the test_labels array.
   
Call the model.fit method to start training. It means "fit" the model to training data.
------------------------------------------------------------------------------------------------------------------------
'''
model.fit(train_images, train_labels, epochs=5)

'''
------------------------------------------------------------------------------------------------------------------------
Evaluation of accuracy rate
Next, compare the performance of the model against the test data set.
------------------------------------------------------------------------------------------------------------------------
'''
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

'''
------------------------------------------------------------------------------------------------------------------------
Predict
Once you have trained a model, you can use it to make classification predictions of images.
------------------------------------------------------------------------------------------------------------------------
'''
predictions = model.predict(test_images)

'''
predict result of test_image[0]

The prediction result is an array of 10 numbers.
This represents the "certainty" that the image falls under each of the 10 clothing types.
'''
print ('predictions[0] = {0}'.format(predictions[0]))

# pridict label of test_image[0]
print ('pridict label of test_image[0] =  {0}'.format(np.argmax(predictions[0])))

# Check the label of actual test_image [0].
print ('real label of test_image[0] = {0}'.format(test_labels[0]))

# You can try graphing all 10 channels.
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


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

'''
note:
Let's display some images in the forecast along with the forecast values. The correct forecast is labeled blue, and the wrong one is labeled red.
The numbers show the percent (percent) of the labels that were predicted.
Keep in mind that it may seem wrong but you may be wrong.
'''
# Display X test images, predicted labels, and correct labels.
# The correct forecast is shown in blue and the wrong forecast is shown in red.

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# Extract one image from the test data set.
img = test_images[0]

print(img.shape)

# Make an image a member of only one batch.
img = (np.expand_dims(img,0))

print(img.shape)

# predict label
predictions_single = model.predict(img)

print('predictions_single = {0}'.format(predictions_single))

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

plt.show()

prediction = predictions[0]

print('rediction label = {0}'.format(np.argmax(prediction)))
