# Common imports
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from PIL import Image
from subprocess import check_call
from matplotlib.colors import ListedColormap


'''
----------------------------------------------------------------------
    Setup
----------------------------------------------------------------------
'''
# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

'''
------------------------------------------------------------------------------------------------------------------------
6 Decision tree
------------------------------------------------------------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          6.1 Decision tree training and visualization                                                \n'
      '------------------------------------------------------------------------------------------------------\n')
# load data
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_fit = tree_clf.fit(X, y)
print('tree_clf_fit = \n{0}\n'.format(tree_clf_fit))

pdp_data = export_graphviz(
        tree_clf,
        out_file=image_path("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

check_call(['dot','-Tpng',image_path("iris_tree.dot"),'-o',image_path("iris_tree.png")])

img = Image.open(image_path("iris_tree.png"))
img.show()

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)

save_fig("decision_tree_decision_boundaries_plot")
plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '          6.3 Estimating class probabilities                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# Estimated probability of iris type when petal length = 5, petal width = 1.5
tree_clf_predict_proba = tree_clf.predict_proba([[5, 1.5]])
print('tree_clf_predict_proba = \n{0}\n'.format(tree_clf_predict_proba))

# Estimated value of iris type when petal length = 5, petal width = 1.5
tree_clf_predict = tree_clf.predict([[5, 1.5]])
print('tree_clf_priedict = {0}\n'.format(tree_clf_predict))

print('------------------------------------------------------------------------------------------------------\n'
      '          Sensitivity to training set details                                                         \n'
      '------------------------------------------------------------------------------------------------------\n')
# to find widest Iris-Versicolor flower
widest_iris_versicolor = X[(X[:, 1]==X[:, 1][y==1].max()) & (y==1)]
print('widest_iris_versicolor = {0}\n'.format(widest_iris_versicolor))

not_widest_versicolor = (X[:, 1]!=1.8) | (y==2)
X_tweaked = X[not_widest_versicolor]
y_tweaked = y[not_widest_versicolor]

tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked_fit = tree_clf_tweaked.fit(X_tweaked, y_tweaked)
