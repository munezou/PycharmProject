# Data gathering
# ----------------------------------
#
# This function gives us the ways to access
# the various data sets we will need

# Data Gathering
import os
import datetime
import requests
import io
import tarfile
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from packaging import version
from tensorflow.python.framework import ops
from sklearn import datasets

tf.compat.v1.disable_eager_execution()
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."


# Iris Data
iris = datasets.load_iris()
print('len(iris.data) = {0}\n'.format(len(iris.data)))
print('len(iris.target) = {0}\n'.format(len(iris.target)))
print('iris.data[0] = {0}\n'.format(iris.data[0]))
print('set(iris.target) = \n{0}\n'.format(set(iris.target)))

# Low Birthrate Data
birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')
birth_header = birth_data[0].split('\t')
birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
print('len(birth_data) = {0}\n'.format(len(birth_data)))
print('len(birth_data[0]) = {0}\n'.format(len(birth_data[0])))


# Housing Price Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print('x_train.shape[0] = {0}\n'.format(x_train.shape[0]))
print('x_train.shape[1] = {0}\n'.format(x_train.shape[1]))


# MNIST Handwriting Data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

one_hot_y_train = tf.one_hot(
    indices=y_train[:],
    depth=10,
    on_value=1.0,
    off_value=0.0,
    axis=-1,
    dtype=tf.float32
)

one_hot_y_test = tf.one_hot(
    indices=y_test[:],
    depth=10,
    on_value=1.0,
    off_value=0.0,
    axis=-1,
    dtype=tf.float32
)

print('len(x_train) = {0}'.format(len(x_train)))
print('len(y_train) = {0}'.format(len(y_train)))
print('len(x_test) = {0}'.format(len(x_test)))
print('len(y_test) = {0}'.format(len(y_test)))
print('y_train[5] = {0}'.format(y_train[5]))
print('one_hot_y_train[5, :] = {0}\n'.format(one_hot_y_train[5, :]))

# CIFAR-10 Image Category Dataset
# The CIFAR-10 data ( https://www.cs.toronto.edu/~kriz/cifar.html ) contains 60,000 32x32 color images of 10 classes.
# It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
# Alex Krizhevsky maintains the page referenced here.
# This is such a common dataset, that there are built in functions in TensorFlow to access this data.

# Running this command requires an internet connection and a few minutes to download all the images.
(X_train, y_train), (X_test, y_test) = tf.compat.v1.keras.datasets.cifar10.load_data()

print('X_train.shape = {0}\n'.format(X_train.shape))
print('y_train.shape = {0}\n'.format(y_train.shape))
print('y_train[0, ] = {0}\n'.format(y_train[0, ]))  # this is a frog

# Plot the 0-th image (a frog)
img = Image.fromarray(X_train[0, :, :, :])
plt.imshow(img)


# Ham/Spam Text Data


# Get/read zip file
zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r = requests.get(zip_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('SMSSpamCollection')
# Format Data
text_data = file.decode()
text_data = text_data.encode('ascii',errors='ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
print('len(text_data_train) = {0}\n'.format(len(text_data_train)))
print('set(text_data_target) = \n{0}\n'.format(set(text_data_target)))
print('text_data_train[1] = {0}\n'.format(text_data_train[1]))


# Movie Review Data

movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)
# Stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
    s = stream_data.read(16384)
    if not s:  
        break
    tmp.write(s)
stream_data.close()
tmp.seek(0)
# Extract tar file
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
# Save pos/neg reviews
pos_data = []
for line in pos:
    pos_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
neg_data = []
for line in neg:
    neg_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
tar_file.close()

print('len(pos_data) = {0}\n'.format(len(pos_data)))
print('len(neg_data) = {0}\n'.format(len(neg_data)))
print('neg_data[0] = {0}\n'.format(neg_data[0]))


# The Works of Shakespeare Data

shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content
# Decode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')
# Drop first few descriptive paragraphs.
shakespeare_text = shakespeare_text[7675:]
print('len(shakespeare_text) = {0}\n'.format(len(shakespeare_text)))


# English-German Sentence Translation Data
sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(sentence_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('deu.txt')

# Format Data
eng_ger_data = file.decode()
eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
eng_ger_data = eng_ger_data.decode().split('\n')
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x) >= 1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
print('len(english_sentence) = {0}\n'.format(len(english_sentence)))
print('len(german_sentence) = {0}\n'.format(len(german_sentence)))
print('eng_ger_data[10] = {0}\n'.format(eng_ger_data[10]))

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished         data_gathering_v2.py                                  ({0})   \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print()
print()
print()