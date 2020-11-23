from __future__ import print_function
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def gen_data(n_samples, test_size=.3):
	print("[src.py] gen_data")
	mean0 = [0., 0.]
	cov0 = [[1., .5], [.5, 1.]]
	data0 = np.random.multivariate_normal(mean0, cov0, n_samples)
	class0 = [0] * n_samples
	mean1 = [2., 2.]
	cov1 = [[1., .5], [.5, 1.]]
	data1 = np.random.multivariate_normal(mean1, cov1, n_samples)
	class1 = [1] * n_samples
	data = np.vstack([data0, data1])
	labels = np.hstack([class0, class1])
	return train_test_split(data, labels, test_size=test_size)


def train(x_train, y_train):
	print("[src.py] train")
	clf = RandomForestClassifier()
	clf.fit(x_train, y_train)
	return clf


def evaluate(clf, x_test, y_test):
	print("[src.py] evaluate")
	pred = clf.predict(x_test)
	confmat = confusion_matrix(y_test, pred)
	return confmat