'''
------------------------------------------------------------------------------------------------------------------------
Exercise:
train a LinearSVC on a linearly separable dataset.
Then train an SVC and a SGDClassifier on the same dataset.
See if you can get them to produce roughly the same model.

Let's use the Iris dataset: the Iris Setosa and Iris Versicolor classes are linearly separable.
------------------------------------------------------------------------------------------------------------------------
'''
# common library
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

print('data information = \n{0}'.format(iris.DESCR))

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

# Extract target = 0 (Setosa) and target = 1 (Versicolour).
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# Compare LinearSVC, SVC and SGDClassifier.
