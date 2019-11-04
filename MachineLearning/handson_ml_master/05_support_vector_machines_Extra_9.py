# common library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print('--------------------------------------------------------------------------------------------------------------\n'
      ' 9.                                                                                                           \n'
      '  train an SVM classifier on the MNIST dataset.                                                               \n'
      '  Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all 10 digits.\n'
      '  You may want to tune the hyperparameters using small validation sets to speed up the process.               \n'
      '  What accuracy can you reach?                                                                                \n'
      '  First, let us load the dataset and split it into a training set and a test set.                             \n'
      '  We could use train_test_split().                                                                            \n'
      '  but people usually just take the first 60,000 instances for the training set,                               \n'
      '  and the last 10,000 instances for the test set                                                              \n'
      ' (this makes it possible to compare your model is performance with others):                                   \n'
      '--------------------------------------------------------------------------------------------------------------\n')
mnist = fetch_openml('mnist_784', version=1, cache=True)

print('data information = \n{0}'.format(mnist.DESCR))

X = mnist["data"]
y = mnist["target"]

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

# Many training algorithms are sensitive to the order of the training instances, so it's generally good practice to shuffle them first:
np.random.seed(42)
rnd_idx = np.random.permutation(60000)
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

# Let's start simple, with a linear SVM classifier.
# It will automatically use the One-vs-All (also called One-vs-the-Rest, OvR) strategy, so there's nothing special we need to do. Easy!
lin_clf = LinearSVC(random_state=42)
lin_clf_fit = lin_clf.fit(X_train, y_train)
print('lin_clf_fit = \n{0}'.format(lin_clf_fit))
print()

# Let's make predictions on the training set and measure the accuracy
# (we don't want to measure it on the test set yet, since we have not selected and trained the final model yet):
y_pred = lin_clf.predict(X_train)
lin_clf_accuracy = accuracy_score(y_train, y_pred)
print('lin_clf_accuracy = {0}'.format(lin_clf_accuracy))
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# Let us start simple, with a linear SVM classifier.
# It will automatically use the One-vs-All (also called One-vs-the-Rest, OvR) strategy,
# so there's nothing special we need to do. Easy!