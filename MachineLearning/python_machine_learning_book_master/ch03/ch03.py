# common library
import os
import warnings
import numpy as np
from PIL import Image
from subprocess import check_call

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score

# to make this notebook's output stable across runs
np.random.seed(42)

'''
------------------------------------------------------------------------------------
Setup
------------------------------------------------------------------------------------
'''
# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, 'Image', fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "Image", fig_id)

# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action='ignore', message='^internal gelsd')

'''
--------------------------------------------------------------------------------
Chapter 3 - A Tour of Machine Learning Classifiers Using Scikit-Learn
--------------------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          3.2.1 Perceptron training with scikit-learn                                                 \n'
      '------------------------------------------------------------------------------------------------------\n')
# First steps with scikit-learn
iris = datasets.load_iris()

# data set information
print('iris data information = \n{0}\n'.format(iris['DESCR']))

X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))

# Splitting data into 70% training and 30% test data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'''
================================================
 Training a perceptron via scikit-learn
================================================
'''
# Redefining the plot_decision_region function from chapter 2:
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

print('y_test.shape = \n{0}\n'.format(y_test.shape))

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

'''
------------------------------------------------------------------
  Declare function
------------------------------------------------------------------
'''

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.6, c=cmap(idx), edgecolor='black', marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, edgecolor='black', linewidths=1, marker='o', s=55, label='test set')

# Training a perceptron model using the standardized training data:
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
save_fig('iris_perceptron_scikit', tight_layout=False)
plt.show()

'''
====================================================================================
 3.3 Modeling class probabilities via logistic regression
====================================================================================
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          3.3.1 Intuitive knowledge and conditional probabilities of logistic regression              \n'
      '------------------------------------------------------------------------------------------------------\n')
# Logistic regression intuition and conditional probabilities
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.title("sigmoid function")
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
save_fig('sigmoid', tight_layout=False)
plt.show()

im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "03_03.png"))
im.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          3.3.2 Logistic function weight learning                                                      \n'
      '------------------------------------------------------------------------------------------------------\n')
def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
save_fig('log_cost', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          3.3.3 Training a logistic regression model with scikit-learn                                \n'
      '------------------------------------------------------------------------------------------------------\n')

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
plt.title("LogisticRegression(C=1000.0)")
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
save_fig('logistic_regression', tight_layout=False)
plt.show()

ir_predict_proba = lr.predict_proba(X_test_std[0, :].reshape(1, -1))
print('ir_predict_proba = \n{0}\n'.format(ir_predict_proba))

print('------------------------------------------------------------------------------------------------------\n'
      '          3.3.4 Tackling overfitting via regularization                                               \n'
      '------------------------------------------------------------------------------------------------------\n')
im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "03_06.png"))
im.show()

weights, params = [], []
for c in np.arange(-5., 5.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
save_fig('regression_path', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          3.4 Maximum margin classification with support vector machines                              \n'
      '------------------------------------------------------------------------------------------------------\n')
im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "03_07.png"))
im.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          3.4.2 Dealing with non-linear separable cases using slack variables                         \n'
      '------------------------------------------------------------------------------------------------------\n')

im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "03_08.png"))
im.show()

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title("SVC(kernel='linear', C=1.0)")
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
save_fig('support_vector_machine_linear', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          3.4.3 Alternative implementations in scikit-learn                                           \n'
      '------------------------------------------------------------------------------------------------------\n')
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title("SVC(kernel='linear', C=1.0)")
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
save_fig('support_vector_machine_linear', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          3.5 Solving non-linear problems using a kernel SVM                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# create raw data
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.title("A simple data set in XOR format")
plt.legend(loc='best')
save_fig('xor', tight_layout=False)
plt.show()

im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "03_11.png"))
im.show()

print('------------------------------------------------------------------------------------------------------\n'
      '     3.5.1 Using the kernel trick to find separating hyperplanes in higher dimensional space          \n'
      '------------------------------------------------------------------------------------------------------\n')
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.title("SVC(kernel='rbf', gamma=0.1, C=10) with using XOR data")
plt.legend(loc='upper left')
save_fig('support_vector_machine_rbf_xor', tight_layout=False)
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title("SVC(kernel='rbf', gamma=0.2, C=1.0) with using iris data")
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
save_fig('support_vector_machine_rbf_iris_1', tight_layout=False)
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,  classifier=svm, test_idx=range(105, 150))
plt.title("SVC(kernel='rbf', gamma=100, C=1.0) with using iris data")
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
save_fig('support_vector_machine_rbf_iris_2', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '     3.6 Decision tree learning                                                                       \n'
      '------------------------------------------------------------------------------------------------------\n')
im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "03_15.png"))
im.show()

print('------------------------------------------------------------------------------------------------------\n'
      '     3.6.1 Maximizing information gain - getting the most bang for the buck                           \n'
      '------------------------------------------------------------------------------------------------------\n')
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy (scaled)',
                           'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.title("Maximizing information gain")
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
save_fig('impurity', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '     3.6.2 Building a decision tree                                                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))

plt.title("a decision tree")
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
save_fig('decision_tree_decision', tight_layout=False)
plt.show()


export_graphviz(
        tree,
        out_file=image_path("tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

check_call(['dot','-Tpng',image_path("tree.dot"),'-o',image_path("tree.png")])

img = Image.open(image_path("tree.png"))
img.show()

print('------------------------------------------------------------------------------------------------------\n'
      '     3.6.3 Combining weak to strong learners via random forests                                       \n'
      '------------------------------------------------------------------------------------------------------\n')
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))

plt.title('Random Forest Classifier')
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
save_fig('random_forest', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '     3.7 K-nearest neighbors - a lazy learning algorithm                                              \n'
      '------------------------------------------------------------------------------------------------------\n')
im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "03_20.png"))
im.show()

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
save_fig('k_nearest_neighbors', tight_layout=False)
plt.show()
