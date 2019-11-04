# common library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

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
lin_clf = LinearSVC(random_state=42)
lin_clf_fit = lin_clf.fit(X_train_scaled, y_train)
print('lin_clf_fit = \n{0}'.format(lin_clf_fit))
print()

y_pred = lin_clf.predict(X_train_scaled)
lin_clf_accuracy = accuracy_score(y_train, y_pred)
print('lin_clf_accuracy = {0}'.format(lin_clf_accuracy))
print()

# That's much better (we cut the error rate in two), but still not great at all for MNIST.
# If we want to use an SVM, we will have to use a kernel. Let's try an SVC with an RBF kernel (the default).
svm_clf = SVC(decision_function_shape="ovr", gamma="auto")
svm_clf_fit = svm_clf.fit(X_train_scaled[:10000], y_train[:10000])

y_pred = svm_clf.predict(X_train_scaled)
accuracy_score(y_train, y_pred)

# That's promising, we get better performance even though we trained the model on 6 times less data.
# Let's tune the hyperparameters by doing a randomized search with cross validation.
# We will do this on a small dataset just to speed up the process:
param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv_fit = rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])
print('rnd_search_cv_fit = \mn{0}'.format(rnd_search_cv_fit))
print()

print('rnd_search_cv.best_estimator_ = {0}'.format(rnd_search_cv.best_estimator_))
print()
print('rnd_search_cv.best_score_ = {0}'.format(rnd_search_cv.best_score_))
print()

# This looks pretty low but remember we only trained the model on 1,000 instances.
# Let's retrain the best estimator on the whole training set (run this at night, it will take hours):
rnd_search_cv_fit = rnd_search_cv.best_estimator_.fit(X_train_scaled, y_train)
print('rnd_search_cv_fit = \n{0}'.format(rnd_search_cv_fit))
print()

y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
y_pred_accuracy_score = accuracy_score(y_train, y_pred)
print('y_pred_accuracy_score = {0}'.format(y_pred_accuracy_score))
print()

# Ah, this looks good! Let's select this model. Now we can test it on the test set:
y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
y_pred_accuracy_score = accuracy_score(y_test, y_pred)
print('y_pred_accuracy_score(test) = {0}'.format(y_pred_accuracy_score))
print()

'''
Not too bad, but apparently the model is overfitting slightly. 
It's tempting to tweak the hyperparameters a bit more (e.g. decreasing C and/or gamma), 
but we would run the risk of overfitting the test set. 
Other people have found that the hyperparameters C=5 and gamma=0.005 yield even better performance (over 98% accuracy). 
By running the randomized search for longer and on a larger part of the training set, you may be able to find this as well.
'''