# Data gathering
#----------------------------------
#
# This function gives us the ways to access
# the various data sets we will need

# Data Gathering
import os
import io
from zipfile import ZipFile
import requests
import tarfile
import input_data
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn import datasets
from keras.datasets import boston_housing
ops.reset_default_graph()

tf.compat.v1.disable_eager_execution()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Iris Data
iris = datasets.load_iris()
print('len(iris.data) = {0}'.format(len(iris.data)))
print('len(iris.target) = {0}'.format(len(iris.target)))
print('iris.data[0] = {0}'.format(iris.data[0]))
print('set(iris.target) = {0}\n'.format(set(iris.target)))

# Low Birthrate Data
birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')
birth_header = birth_data[0].split('\t')
birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
print('len(birth_data) = {0}'.format(len(birth_data)))
print('len(birth_data[0]) = {0}\n'.format(len(birth_data[0])))

# Housing Price Data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print('x_train.shape[0] = {0}'.format(x_train.shape[0]))
print('x_train.shape[1] = {0}\n'.format(x_train.shape[1]))

# MNIST Handwriting Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('len(mnist.train.images) = {0}'.format(len(mnist.train.images)))
print('len(mnist.test.images) = {0}'.format(len(mnist.test.images)))
print(''.format(len(mnist.validation.images)))
print('mnist.train.labels[1, :] = {0}\n'.format(mnist.train.labels[1, :]))

# CIFAR-10 Image Category Dataset
# The CIFAR-10 data ( https://www.cs.toronto.edu/~kriz/cifar.html ) contains 60,000 32x32 color images of 10 classes.
# It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
# Alex Krizhevsky maintains the page referenced here.
# This is such a common dataset, that there are built in functions in TensorFlow to access this data.

# Running this command requires an internet connection and a few minutes to download all the images.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print('X_train.shape = {0}'.format(X_train.shape))
print('y_train.shape = {0}'.format(y_train.shape))
print(''.format(y_train[0, ]))  # this is a frog

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
print('len(text_data_train) = {0}'.format(len(text_data_train)))
print('set(text_data_target) = {0}'.format(set(text_data_target)))
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
    pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
neg_data = []
for line in neg:
    neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
tar_file.close()

print('len(pos_data) = {0}'.format(len(pos_data)))
print('len(neg_data) = {0}'.format(len(neg_data)))
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
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]

print('len(english_sentence) = {0}'.format(len(english_sentence)))
print('len(german_sentence) = {0}'.format(len(german_sentence)))
print('eng_ger_data[10] = {0}\n'.format(eng_ger_data[10]))
