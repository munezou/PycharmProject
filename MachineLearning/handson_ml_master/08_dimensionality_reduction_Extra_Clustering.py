# common liblary
import time
import numpy as np
import os
import warnings


import matplotlib as mpl
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs


from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans


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
CHAPTER_ID = "unsupervised_learning"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")

'''
------------------------------------------------------------------------------------------------------------
Extra Material – Clustering
------------------------------------------------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          Introduction - Classification vs Clustering                                                 \n'
      '------------------------------------------------------------------------------------------------------\n')
# load iris data
data = load_iris()

# input data and target
X = data['data']
y = data['target']

print('data["target_names"] = \n{0}\n'.format(data['target_names']))

# plot iris data
plt.figure(figsize=(9, 3.5))

plt.subplot(121)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris-Setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris-Versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris-Virginica")
plt.title("Classification")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(122)
plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
plt.title("Clustering")
plt.xlabel("Petal length", fontsize=14)
plt.tick_params(labelleft=False)

save_fig("classification_vs_clustering_diagram")
plt.show()
print()

'''
------------------------------------------------------------------------------------------------
A Gaussian mixture model (explained below) can actually separate these clusters pretty well 
(using all 4 features: petal length & width, and sepal length & width).
------------------------------------------------------------------------------------------------
'''
y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
mapping = np.array([2, 0, 1])
y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

plt.plot(X[y_pred==0, 2], X[y_pred==0, 3], "yo", label="Cluster 1")
plt.plot(X[y_pred==1, 2], X[y_pred==1, 3], "bs", label="Cluster 2")
plt.plot(X[y_pred==2, 2], X[y_pred==2, 3], "g^", label="Cluster 3")
plt.title("predict by GaussianMixture")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.show()

# Number of matches between forecast and actual data
number_matches = np.sum(y_pred == y)
print('number_maches = {0}\n'.format(number_matches))

# Accuracy rate
accuracy_rate = np.sum(y_pred==y) / len(y_pred)
print('accuracy_rate = {0}\n'.format(accuracy_rate))

print('------------------------------------------------------------------------------------------------------\n'
      '          K-Means                                                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')
# Let's start by generating some blobs:
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

# create data
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

# Now let's plot them:
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.figure(figsize=(8, 4))
plt.title("raw data by make_blobs()")
plot_clusters(X)
save_fig("blobs_diagram")
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          Fit and Predict by KMeans                                                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
# Let's train a K-Means clusterer on this dataset.
# It will try to find each blob's center and assign each instance to the closest blob:

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

# Each instance was assigned to one of the 5 clusters:
print('y_pred = \n{0}\n'.format(y_pred))

#
print('y_pred is kmeans.labels_ = {0}\n'.format(y_pred is kmeans.labels_))

# And the following 5 centroids (i.e., cluster centers) were estimated:
kmeans_cluster_center = kmeans.cluster_centers_
print('kmeans.cluster_centers_ = \n{0}\n'.format(kmeans.cluster_centers_))

'''
Note that the KMeans instance preserves the labels of the instances it was trained on. 
Somewhat confusingly, 
in this context, the label of an instance is the index of the cluster that instance gets assigned to:
'''
kmeans_labels = kmeans.labels_
print('kmeans_labels = \n{0}\n'.format(kmeans_labels))

# Of course, we can predict the labels of new instances:
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])

# predict by kmeans
kmeans_predict = kmeans.predict(X_new)
print('kmeans_predict = \n{0}\n'.format(kmeans_predict))

print('------------------------------------------------------------------------------------------------------\n'
      '          Decision Boundaries                                                                         \n'
      '------------------------------------------------------------------------------------------------------\n')
# Let's plot the model's decision boundaries. This gives us a Voronoi diagram:
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

plt.figure(figsize=(8, 4))
plt.title("vornoi diagram")
plot_decision_boundaries(kmeans, X)
save_fig("voronoi_diagram")
plt.show()
print()

'''
Not bad! 
Some of the instances near the edges were probably assigned to the wrong cluster, 
but overall it looks pretty good.
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          Hard Clustering vs Soft Clustering                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')

'''
-------------------------------------------------------------------------------------------------------------
Rather than arbitrarily choosing the closest cluster for each instance, which is called hard clustering, 
it might be better measure the distance of each instance to all 5 centroids. 
This is what the transform() method does:
-------------------------------------------------------------------------------------------------------------
'''
kmeans_transform = kmeans.transform(X_new)
print('kmeans_transform = \n{0}\n'.format(kmeans_transform))

# You can verify that this is indeed the Euclidian distance between each instance and each centroid:
euclidian_distance = np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2)
print('euclidian_distance = \n{0}\n'.format(euclidian_distance))

print('kmeans_transform == euclidian_distance = {0}\n'.format(kmeans_transform == euclidian_distance))

print('------------------------------------------------------------------------------------------------------\n'
      '          K-Means Algorithm                                                                           \n'
      '------------------------------------------------------------------------------------------------------\n')

'''
---------------------------------------------------------------------------------------------------------------
The K-Means algorithm is one of the fastest clustering algorithms, but also one of the simplest:

* First initialize  𝑘  centroids randomly:  
    𝑘  distinct instances are chosen randomly from the dataset and the centroids are placed at their locations.
* Repeat until convergence (i.e., until the centroids stop moving):
    * Assign each instance to the closest centroid.
    * Update the centroids to be the mean of the instances that are assigned to them.

The KMeans class applies an optimized algorithm by default. 
To get the original K-Means algorithm (for educational purposes only), you must set init="random", n_init=1and algorithm="full". 
These hyperparameters will be explained below.

Let's run the K-Means algorithm for 1, 2 and 3 iterations, to see how the centroids move around:
---------------------------------------------------------------------------------------------------------------
'''
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=1, random_state=1)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=2, random_state=1)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=3, random_state=1)
kmeans_iter1_fit = kmeans_iter1.fit(X)
kmeans_iter2_fit = kmeans_iter2.fit(X)
kmeans_iter3_fit = kmeans_iter3.fit(X)

print('kmeans_iter1_fit = \n{0}\n'.format(kmeans_iter1_fit))
print('kmeans_iter2_fit = \n{0}\n'.format(kmeans_iter2_fit))
print('kmeans_iter3_fit = \n{0}\n'.format(kmeans_iter3_fit))

# And let's plot this:
plt.figure(figsize=(10, 8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)", fontsize=14)

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Label the instances", fontsize=14)

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

save_fig("kmeans_algorithm_diagram")
plt.show()