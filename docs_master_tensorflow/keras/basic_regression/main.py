import pathlib
import os
import platform

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

'''
------------------------------------------------------------------------------------------------------------------------
Acquisition of data
------------------------------------------------------------------------------------------------------------------------
'''
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
data_path = os.path.join(PROJECT_ROOT_DIR, "AI_data", "auto-mpg.data")
raw_dataset = pd.read_csv(data_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(dataset.tail())

print()

'''
------------------------------------------------------------------------------------------------------------------------
Data cleansing
------------------------------------------------------------------------------------------------------------------------
'''
print('dataset.isna().sum() = \n{0}'.format(dataset.isna().sum()))

'''
We will remove these lines for simplicity in this first tutorial.
dropna() : Return ExtensionArray without NA values
'''
dataset = dataset.dropna()

print('--- < confirm whether na exist or not. >---')
# confirm whether na exist or not.
print(dataset.isna().sum())

print()

'''
------------------------------------------------------------------------------------------------------------------------
The "Origin" column is a category, not a number. Because of this, one hot encoding is done.

one hot encoding:
  One-Hot, that is, one vector is a one (1) and the other is zero (vector).
  In economics and statistics, it is sometimes called "dummy variable".
------------------------------------------------------------------------------------------------------------------------
'''
origin = dataset.pop('Origin')
print(origin.tail())

print ('---< origin column is changed by one hot en-cording. >---')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print ('origin column after one hot en-cording = \n{0}'.format(dataset.tail()))

print()

'''
------------------------------------------------------------------------------------------------------------------------
Divide the data into training and testing sets.
------------------------------------------------------------------------------------------------------------------------
'''
print('---< Divide the data into training and testing sets. >---')

train_dataset = dataset.sample(frac=0.8,random_state=0)
print ('train_dataset = \n{0}'.format(train_dataset.tail()))
test_dataset = dataset.drop(train_dataset.index)
print ('test_dataset = \n{0}'.format(test_dataset.tail()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Survey of data
------------------------------------------------------------------------------------------------------------------------
'''
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# Overall statistics
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print('Overall statistics = \n{0}'.format(train_stats))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Separation of labels and features
------------------------------------------------------------------------------------------------------------------------
'''
print ('---< Separation of labels and features >---')
train_labels = train_dataset.pop('MPG')
print ('train_labels = \n{0}'.format(train_labels.tail()))
test_labels = test_dataset.pop('MPG')
print ('test_labels = \n{0}'.format(test_labels.tail()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Data normalization
------------------------------------------------------------------------------------------------------------------------
'''
print ('---< Data normalization >---')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
print ('normed_train_data = \n{0}'.format(normed_train_data.tail()))
normed_test_data = norm(test_dataset)
print ('normed_test_data = \n{0}'.format(normed_test_data.tail()))
print()


sns.pairplot(normed_test_data[["Cylinders", "Displacement", "Weight", 'Acceleration', 'Model Year']], diag_kind="kde")
plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
Build a model:
Let's build a model. 
Here we use the Sequential model, which consists of a hidden layer of two full bonds and an output layer that returns one continuous value.
The steps to build a model are included in one function called build_model. 
It is to build the second model afterward.
------------------------------------------------------------------------------------------------------------------------
'''
print('---< Build a model >---')
print('len(train_dataset.keys()) = {0}'.format(len(train_dataset.keys())))
print('train_dataset.keys() = {0}'.format(train_dataset.keys()))
print()

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile (
                loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error']
                )
  
  return model

# model build
model = build_model()

'''
------------------------------------------------------------------------------------------------------------------------
model verification
------------------------------------------------------------------------------------------------------------------------
'''
print('model verification = \n{0}'.format(model.summary()))
print()

# Let's try a model. Take a batch of 10 samples of training data and use it to call the model.predict method.
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print('example_result = \n{0}'.format(example_result))
print ()

'''
------------------------------------------------------------------------------------------------------------------------
Model training
Train a model for 1000 epochs and record training and verification accuracy rates in the history object.
------------------------------------------------------------------------------------------------------------------------
'''
print ('---< model training >---')
# The progress is displayed by outputting one dot each time the epoch ends.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('hist = \n{0}'.format(hist.tail()))
print()

# Output Graphic
import matplotlib.pyplot as plt

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])

plot_history(history)
plt.show()

print ('---< model modify >---')
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
plt.show()

'''
Model evaluation results:
Let's use a test data set not used to train the model to see how well it generalizes the model. 
This will show you how accurately the model can be predicted in the real world.
'''
print('---< Model evaluation results >---')
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

'''
------------------------------------------------------------------------------------------------------------------------
Prediction using a model
Finally, use test data to predict MPG values.
------------------------------------------------------------------------------------------------------------------------
'''
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
Let's look at the distribution of the error.
------------------------------------------------------------------------------------------------------------------------
'''
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()