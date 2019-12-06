# common liblary
import time
import numpy as np
from scipy.stats import norm
import os
import warnings
import timeit

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.image import imread
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.datasets import make_moons

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.mixture import GaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

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

print('------------------------------------------------------------------------------------------------------\n'
      '          K-Means Variability                                                                         \n'
      '------------------------------------------------------------------------------------------------------\n')
'''
-------------------------------------------------------------------------------------------------------------
In the original K-Means algorithm, 
the centroids are just initialized randomly, 
and the algorithm simply runs a single iteration to gradually improve the centroids, as we saw above.

However, one major problem with this approach is that if you run K-Means multiple times 
(or with different random seeds), it can converge to very different solutions, as you can see below:
-------------------------------------------------------------------------------------------------------------
'''


def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", random_state=11)
kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", random_state=19)

plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X, "Solution 1",
                          "Solution 2 (with a different random init)")

save_fig("kmeans_variability_diagram")
plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '          Inertia (The square error within each cluster)                                              \n'
      '------------------------------------------------------------------------------------------------------\n')

'''
-------------------------------------------------------------------------------------------------------------
To select the best model, 
e will need a way to evaluate a K-Mean model's performance. Unfortunately, 
clustering is an unsupervised task, so we do not have the targets. 
But at least we can measure the distance between each instance and its centroid. 
This is the idea behind the inertia metric:
-------------------------------------------------------------------------------------------------------------
'''

kmeans_inertia = kmeans.inertia_
print('kmeans_inertia = {0}\n'.format(kmeans_inertia))

'''
As you can easily verify, 
inertia is the sum of the squared distances between each training instance and its closest centroid:
'''

X_dist = kmeans.transform(X)
kmeans_inertia_another = np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_] ** 2)
print('kmeans_inertia_another = {0}\n'.format(kmeans_inertia_another))

'''
---------------------------------------------------------------------------------------------------------------
The score() method returns the negative inertia. 
Why negative? Well, it is because a predictor's score() method must always respect the "great is better" rule.
---------------------------------------------------------------------------------------------------------------
'''
kmeans_score = kmeans.score(X)
print('kmeans_score = {0}\n'.format(kmeans_score))

print('------------------------------------------------------------------------------------------------------\n'
      '          Multiple Initializations                                                                    \n'
      '------------------------------------------------------------------------------------------------------\n')

'''
---------------------------------------------------------------------------------------------------------------
So one approach to solve the variability issue is to simply run the K-Means algorithm multiple times with different random initializations, 
and select the solution that minimizes the inertia. 
For example, here are the inertias of the two "bad" models shown in the previous figure:
---------------------------------------------------------------------------------------------------------------
'''
print('kmeans_rnd_init1.inertia_ = {0}\n'.format(kmeans_rnd_init1.inertia_))
print('kmeans_rnd_init2.inertia_ = {0}\n'.format(kmeans_rnd_init2.inertia_))

'''
---------------------------------------------------------------------------------------------------------------
As you can see, they have a higher inertia than the first "good" model we trained, 
which means they are probably worse.

When you set the n_init hyperparameter, 
Scikit-Learn runs the original algorithm n_init times, 
and selects the solution that minimizes the inertia. By default, Scikit-Learn sets n_init=10.
---------------------------------------------------------------------------------------------------------------
'''
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10, algorithm="full", random_state=11)

kmeans_rnd_10_inits_fit = kmeans_rnd_10_inits.fit(X)
print('kmeans_rnd_10_inits_fit = \n{0}\n'.format(kmeans_rnd_10_inits_fit))

'''
---------------------------------------------------------------------------------------------------------------
As you can see, 
we end up with the initial model, which is certainly the optimal K-Means solution 
(at least in terms of inertia, and assuming  𝑘=5 ).
---------------------------------------------------------------------------------------------------------------
'''
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans_rnd_10_inits, X)
plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '          K-Means++                                                                                   \n'
      '------------------------------------------------------------------------------------------------------\n')

'''
--------------------------------------------------------------------------------------------------------------
Instead of initializing the centroids entirely randomly, 
it is preferable to initialize them using the following algorithm, 
proposed in a 2006 paper by David Arthur and Sergei Vassilvitskii:

* Take one centroid  𝑐1 , chosen uniformly at random from the dataset.
* Take a new center  𝑐𝑖 , choosing an instance  𝐱𝑖  with probability:  
    𝐷(𝐱𝑖)2  /  ∑𝑗=1𝑚𝐷(𝐱𝑗)2  where  𝐷(𝐱𝑖)  is the distance between the instance  𝐱𝑖  and the closest centroid that was already chosen. 
    This probability distribution ensures that instances that are further away from already chosen centroids are much more likely be selected as centroids.
* Repeat the previous step until all  𝑘  centroids have been chosen.

The rest of the K-Means++ algorithm is just regular K-Means. 
With this initialization, the K-Means algorithm is much less likely to converge to a suboptimal solution, 
so it is possible to reduce n_init considerably. Most of the time, this largely compensates for the additional complexity of the initialization process.

To set the initialization to K-Means++, simply set init="k-means++" (this is actually the default):
-------------------------------------------------------------------------------------------------------------
'''
print('KMeans() = \n{0}\n'.format(KMeans()))

good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
kmeans_inertia = kmeans.inertia_
print('kmeans_inertia = {0}\n'.format(kmeans_inertia))

print('------------------------------------------------------------------------------------------------------\n'
      '          Accelerated K-Means                                                                         \n'
      '------------------------------------------------------------------------------------------------------\n')

'''
---------------------------------------------------------------------------------------------------------------
The K-Means algorithm can be significantly accelerated by avoiding many unnecessary distance calculations: 
this is achieved by exploiting the triangle inequality 
(given three points A, B and C, the distance AC is always such that AC ≤ AB + BC) 
and by keeping track of lower and upper bounds for distances between instances and centroids 
(see this 2003 paper by Charles Elkan for more details).

To use Elkan's variant of K-Means, just set algorithm="elkan". 
Note that it does not support sparse data, so by default, Scikit-Learn uses "elkan" for dense data, 
and "full" (the regular K-Means algorithm) for sparse data.
---------------------------------------------------------------------------------------------------------------
'''
result = timeit.timeit('KMeans(algorithm="elkan").fit(X)', globals=globals(), number=50)
print('one time: {0}[sec]\n'.format(result / 50))

result = timeit.timeit('KMeans(algorithm="full").fit(X)', globals=globals(), number=50)
print('one time: {0}[sec]\n'.format(result / 50))

print('------------------------------------------------------------------------------------------------------\n'
      '          Mini-Batch K-Means                                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# Scikit-Learn also implements a variant of the K-Means algorithm that supports mini-batches (see this paper):
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, compute_labels=True, random_state=42)
minibatch_kmeans_fit = minibatch_kmeans.fit(X)
print('minibatch_kmeans_fit = \n{0}\n'.format(minibatch_kmeans_fit))

minibatch_kmeans_inertia = minibatch_kmeans.inertia_
print('minibatch_kmeans_inertia = {0}\n'.format(minibatch_kmeans_inertia))

# If the dataset does not fit in memory, the simplest option is to use the memmap class, just like we did for incremental PCA:
filename = "my_mnist.data"
m, n = 50000, 28 * 28
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
minibatch_kmeans_fit_memmap = minibatch_kmeans.fit(X_mm)
print('minibatch_kmeans_fit_memmap = \n{0}\n'.format(minibatch_kmeans_fit_memmap))

'''
--------------------------------------------------------------------------------------------------------------
If your data is so large that you cannot use memmap, 
things get more complicated. Let's start by writing a function to load the next batch 
(in real life, you would load the data from disk):
--------------------------------------------------------------------------------------------------------------
'''


def load_next_batch(batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]


'''
--------------------------------------------------------------------------------------------------------------
Now we can train the model by feeding it one batch at a time. 
We also need to implement multiple initializations and keep the model with the lowest inertia:
--------------------------------------------------------------------------------------------------------------
'''
np.random.seed(42)

k = 5
n_init = 10
n_iterations = 100
batch_size = 100
init_size = 500  # more data for K-Means++ initialization
evaluate_on_last_n_iters = 10

best_kmeans = None

for init in range(n_init):
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
    X_init = load_next_batch(init_size)
    minibatch_kmeans.partial_fit(X_init)

    minibatch_kmeans.sum_inertia_ = 0
    for iteration in range(n_iterations):
        X_batch = load_next_batch(batch_size)
        minibatch_kmeans.partial_fit(X_batch)
        if iteration >= n_iterations - evaluate_on_last_n_iters:
            minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_

    if (best_kmeans is None or
            minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):
        best_kmeans = minibatch_kmeans

best_kmeans_score = best_kmeans.score(X)
print('best_kmeans_score = {0}\n'.format(best_kmeans_score))

# Mini-batch K-Means is much faster than regular K-Means:
result = timeit.timeit('KMeans(n_clusters=5).fit(X)', globals=globals(), number=50)
print('one time(KMeans): {0}[sec]\n'.format(result / 50))

result = timeit.timeit('MiniBatchKMeans(n_clusters=5).fit(X)', globals=globals(), number=50)
print('one time(MiniBatchKMeans): {0}[sec]\n'.format(result / 50))

'''
----------------------------------------------------------------------------------------------
That's much faster! 
However, its performance is often lower (higher inertia), and it keeps degrading as k increases.
Let's plot the inertia ratio and the training time ratio between Mini-batch K-Means and regular K-Means:
----------------------------------------------------------------------------------------------
'''
times = np.empty((100, 2))
inertias = np.empty((100, 2))
for k in range(1, 101):
    kmeans = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    print("\r{}/{}".format(k, 100), end="")
    times[k - 1, 0] = timeit.timeit('kmeans.fit(X)', globals=globals(), number=10)
    times[k - 1, 1] = timeit.timeit('minibatch_kmeans.fit(X)', globals=globals(), number=10)
    inertias[k - 1, 0] = kmeans.inertia_
    inertias[k - 1, 1] = minibatch_kmeans.inertia_

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
# plt.ylabel("Inertia", fontsize=14)
plt.title("Inertia", fontsize=14)
plt.legend(fontsize=14)
plt.axis([1, 100, 0, 100])

plt.subplot(122)
plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
# plt.ylabel("Training time (seconds)", fontsize=14)
plt.title("Training time (seconds)", fontsize=14)
plt.axis([1, 100, 0, 6])
# plt.legend(fontsize=14)

save_fig("minibatch_kmeans_vs_kmeans")
plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '          Finding the optimal number of clusters                                                      \n'
      '------------------------------------------------------------------------------------------------------\n')
# What if the number of clusters was set to a lower or greater value than 5?
kmeans_k3 = KMeans(n_clusters=3, random_state=42)
kmeans_k8 = KMeans(n_clusters=8, random_state=42)

plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")
save_fig("bad_n_clusters_diagram")
plt.show()
print()

print('kmeans_k3.inertia_ = {0}\n'.format(kmeans_k3.inertia_))
print('kmeans_k8.inertia_ = {0}\n'.format(kmeans_k8.inertia_))

'''
-------------------------------------------------------------------------------------------------------------
No, we cannot simply take the value of  𝑘  that minimizes the inertia, 
since it keeps getting lower as we increase  𝑘 . 
Indeed, the more clusters there are, the closer each instance will be to its closest centroid, 
and therefore the lower the inertia will be. 
However, we can plot the inertia as a function of  𝑘  and analyze the resulting curve:
-------------------------------------------------------------------------------------------------------------
'''
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(4, inertias[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
             )
plt.axis([1, 8.5, 0, 1300])
save_fig("inertia_vs_k_diagram")
plt.show()
print()

'''
------------------------------------------------------------------------------------------------------------
As you can see, there is an elbow at  𝑘=4 , which means that less clusters than that would be bad, 
and more clusters would not help much and might cut clusters in half. 
So  𝑘=4  is a pretty good choice. 
Of course in this example it is not perfect since it means that the two blobs in the lower left will be considered as just a single cluster, 
but it's a pretty good clustering nonetheless.
------------------------------------------------------------------------------------------------------------
'''
plot_decision_boundaries(kmeans_per_k[4 - 1], X)
plt.show()
print()

'''
-----------------------------------------------------------------------------------------------------------
Another approach is to look at the silhouette score, which is the mean silhouette coefficient over all the instances. 
An instance's silhouette coefficient is equal to  (𝑏−𝑎)/max(𝑎,𝑏)  
where  𝑎  is the mean distance to the other instances in the same cluster (it is the mean intra-cluster distance), 
and  𝑏  is the mean nearest-cluster distance, that is the mean distance to the instances of the next closest cluster 
(defined as the one that minimizes  𝑏 , excluding the instance's own cluster). 
The silhouette coefficient can vary between -1 and +1: 
a coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters,
while a coefficient close to 0 means that it is close to a cluster boundary, 
and finally a coefficient close to -1 means that the instance may have been assigned to the wrong cluster.
-----------------------------------------------------------------------------------------------------------
'''
# Let's plot the silhouette score as a function of  𝑘 :
print('silhouette_score(X, kmeans.labels_) = {0}\n'.format(silhouette_score(X, kmeans.labels_)))

silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.title("silhouette score vs k_diagram")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
save_fig("silhouette_score_vs_k_diagram")
plt.show()

'''
---------------------------------------------------------------------------------------------------------------
As you can see, 
this visualization is much richer than the previous one: in particular, although it confirms that  𝑘=4  is a very good choice, 
but it also underlines the fact that  𝑘=5  is quite good as well.
---------------------------------------------------------------------------------------------------------------
'''
'''
---------------------------------------------------------------------------------------------------------------
An even more informative visualization is given when you plot every instance's silhouette coefficient, 
sorted by the cluster they are assigned to and by the value of the coefficient. 
This is called a silhouette diagram:
---------------------------------------------------------------------------------------------------------------
'''
plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")

    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

save_fig("silhouette_analysis_diagram")
plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '          Limits of K-Means                                                                           \n'
      '------------------------------------------------------------------------------------------------------\n')
# Create raw data
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
plt.figure(figsize=(8, 6))
plt.title("phase 1")
plt.scatter(X1[:, 0], X1[:, 1], c='pink')
plt.xlabel("X1[0]")
plt.ylabel("X1[1]")
plt.grid(True)
plt.show()

X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
plt.figure(figsize=(8, 6))
plt.title("phase 2")
plt.scatter(X1[:, 0], X1[:, 1], c='pink')
plt.xlabel("X1[0]")
plt.ylabel("X1[1]")
plt.grid(True)
plt.show()

X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
plt.figure(figsize=(8, 6))
plt.title("phase 3")
plt.scatter(X2[:, 0], X2[:, 1], c='pink')
plt.xlabel("X2[0]")
plt.ylabel("X2[1]")
plt.grid(True)
plt.show()

X2 = X2 + [6, -8]
plt.figure(figsize=(8, 6))
plt.title("phase 4")
plt.scatter(X2[:, 0], X2[:, 1], c='pink')
plt.xlabel("X2[0]")
plt.ylabel("X2[1]")
plt.grid(True)
plt.show()

X = np.r_[X1, X2]
y = np.r_[y1, y2]

plt.figure(figsize=(8, 6))
plot_clusters(X)
plt.show()

# sart clusering
kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)
kmeans_bad = KMeans(n_clusters=3, random_state=42)
kmeans_good_fit = kmeans_good.fit(X)
kmeans_bad_fit = kmeans_bad.fit(X)

print('kmeans_good_fit = \n{0}\n'.format(kmeans_good_fit))
print('kmeans_bad_fit = \n{0}\n'.format(kmeans_bad_fit))

# plot clustering
plt.figure(figsize=(10, 3.2))

plt.subplot(121)
plot_decision_boundaries(kmeans_good, X)
plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)

plt.subplot(122)
plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)
plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)

save_fig("bad_kmeans_diagram")
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          Using clustering for image segmentation                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')
image = imread(os.path.join(PROJECT_ROOT_DIR, "images","unsupervised_learning","ladybug.png"))
print('image.shape = {0}\n'.format(image.shape))

X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

save_fig('image_segmentation_diagram', tight_layout=False)
plt.show()

print()

print('------------------------------------------------------------------------------------------------------\n'
      '          Using Clustering for Preprocessing                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# Let's tackle the digits dataset which is a simple MNIST-like dataset containing 1,797 grayscale 8×8 images representing digits 0 to 9.
X_digits, y_digits = load_digits(return_X_y=True)

# Let's split it into a training set and a test set: test_size(default)=0.25
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

# Now let's fit a Logistic Regression model and evaluate it on the test set:
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg_fit = log_reg.fit(X_train, y_train)

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
--------------------------------------------------------------------------------------------------------------
Okay, that's our baseline: 96.7% accuracy. 
Let's see if we can do better by using K-Means as a preprocessing step. 
We will create a pipeline that will first cluster the training set into 50 clusters 
and replace the images with their distances to the 50 clusters, then apply a logistic regression model:
--------------------------------------------------------------------------------------------------------------
'''
pipeline = Pipeline([
                    ("kmeans", KMeans(n_clusters=50, random_state=42)),
                    ("log_reg", LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)),
                    ])

pipeline_fit = pipeline.fit(X_train, y_train)
print('pipeline_fit = \n{0}\n'.format(pipeline_fit))

pipeline_score = pipeline.score(X_test, y_test)
print('pipeline_score = {0}\n'.format(pipeline_score))

error_rate = 1 - (1 - 0.9822222) / (1 - 0.9666666)
print('error_rate = {0}\n'.format(error_rate))

'''
-------------------------------------------------------------------------------------------------------------
 How about that? 
 We almost divided the error rate by a factor of 2!
 But we chose the number of clusters  𝑘  completely arbitrarily, we can surely do better. 
 Since K-Means is just a preprocessing step in a classification pipeline, 
finding a good value for  𝑘  is much simpler than earlier: 

 there's no need to perform silhouette analysis or minimize the inertia, 
the best value of  𝑘  is simply the one that results in the best classification performance.
-------------------------------------------------------------------------------------------------------------
'''
param_grid = dict(kmeans__n_clusters=range(2, 100))

grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)

print()
print()
grid_clf_fit = grid_clf.fit(X_train, y_train)
print('grid_clf = \n{0}\n'.format(grid_clf_fit))

grid_clf_best_parametrer = grid_clf.best_params_
print('grid_clf_best_parametrer = {0}\n'.format(grid_clf_best_parametrer))

grid_clf_score = grid_clf.score(X_test, y_test)
print('grid_clf_score = {0}\n'.format(grid_clf_score))

'''
-------------------------------------------------------------------------------------------------------------
The performance is slightly improved when  𝑘=90 , so 90 it is.
-------------------------------------------------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          Clustering for Semi-supervised Learning                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')
'''
---------------------------------------------------------------------------------------------------------------
Another use case for clustering is in semi-supervised learning, 
when we have plenty of unlabeled instances and very few labeled instances.

Let's look at the performance of a logistic regression model when we only have 50 labeled instances:
---------------------------------------------------------------------------------------------------------------
'''
n_labeled = 50

log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42, n_jobs=-1)

log_reg_fit = log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
----------------------------------------------------------------------------------------------------------------
It's much less than earlier of course. 
Let's see how we can do better. 
First, let's cluster the training set into 50 clusters, 
then for each cluster let's find the image closest to the centroid. 
We will call these images the representative images:
----------------------------------------------------------------------------------------------------------------
'''
k = 50

kmeans = KMeans(n_clusters=k, random_state=42, n_jobs=-1)

X_digits_dist = kmeans.fit_transform(X_train)

representative_digit_idx = np.argmin(X_digits_dist, axis=0)

X_representative_digits = X_train[representative_digit_idx]

'''
----------------------------------------------------------------------------------------------------------------
Now let's plot these representative images and label them manually:
----------------------------------------------------------------------------------------------------------------
'''
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

save_fig("representative_images_diagram", tight_layout=False)
plt.show()

y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1
])

'''
------------------------------------------------------------------------------------------------------------------
Now we have a dataset with just 50 labeled instances, 
but instead of being completely random instances, each of them is a representative image of its cluster. 
Let's see if the performance is any better:
------------------------------------------------------------------------------------------------------------------
'''
# using classifier is Logistic Regression.
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42, n_jobs=-1)

# fit represent digits
log_reg_fit = log_reg.fit(X_representative_digits, y_representative_digits)
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

# Evaluate the score.
log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
-----------------------------------------------------------------------------------------------------------------
Wow! We jumped from 82.7% accuracy to 92.4%, although we are still only training the model on 50 instances. 
Since it's often costly and painful to label instances, especially when it has to be done manually by experts, 
it's a good idea to make them label representative instances rather than just random instances.

But perhaps we can go one step further: what if we propagated the labels to all the other instances in the same cluster?
-----------------------------------------------------------------------------------------------------------------
'''
# From X_train data, extract the one where the index of kmeans.labels matches the index of y_representative_digits.
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

# using classifier is Logistic Regression.
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42, n_jobs=-1)

# fitting
log_reg_fit = log_reg.fit(X_train, y_train_propagated)
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

# check score
log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
--------------------------------------------------------------------------------------------------------------------
We got a tiny little accuracy boost. 
Better than nothing, but we should probably have propagated the labels only to the instances closest to the centroid, 
because by propagating to the full cluster, we have certainly included some outliers. 
--------------------------------------------------------------------------------------------------------------------
'''

# Let's only propagate the labels to the 20th percentile closest to the centroid:
percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

# using classifier is Logistic Regression.
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42, n_jobs=-1)

log_reg_fit = log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
---------------------------------------------------------------------------------------------------------------------
Nice! With just 50 labeled instances (just 5 examples per class on average!), 
we got 94.2% performance, which is pretty close to the performance of logistic regression on the fully labeled digits dataset (which was 96.7%).
---------------------------------------------------------------------------------------------------------------------
'''

# This is because the propagated labels are actually pretty good: their accuracy is very close to 99%:
identity_ratio = np.mean(y_train_partially_propagated == y_train[partially_propagated])
print('identity_ratio = {0}\n'.format(identity_ratio))

'''
-----------------------------------------------------------------------------------------------------------------------
You could now do a few iterations of active learning:

1. Manually label the instances that the classifier is least sure about, if possible by picking them in distinct clusters.
2 Train a new model with these additional labels.
-----------------------------------------------------------------------------------------------------------------------
'''

print(
    '------------------------------------------------------------------------------------------------------\n'
    '          DBSCAN                                                                                      \n'
    '------------------------------------------------------------------------------------------------------\n'
)
# create make_moons
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# Draw scatter grapnic
plt.figure(figsize=(8, 6))
plt.title("raw data for make_moons(noise=0.05)")
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c='red', label="X0")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c='blue', label="X1")
plt.grid(True)
plt.legend()
plt.show()

# using clustering is DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=5, n_jobs=-1)

# fiiting
dbscan_fit = dbscan.fit(X)
print('dbscan_fit = \n{0}\n'.format(dbscan_fit))

# output label from index = 0 to index = 10
dbscan_label = dbscan.labels_[:10]
print('dbscan_label = \n{0}\n'.format(dbscan_label))

print('len(dbscan.core_sample_indices_) = {0}\n'.format(len(dbscan.core_sample_indices_)))

print('dbscan.core_sample_indices_[:10] = \n{0}\n'.format(dbscan.core_sample_indices_[:10]))

print('dbscan.components_[:3] = \n{0}\n'.format(dbscan.components_[:3]))

unique_labels = np.unique(dbscan.labels_)
print('unique_labels = \n{0}\n'.format(unique_labels))

dbscan2 = DBSCAN(eps=0.2, n_jobs=-1)

dbscan2_fit = dbscan2.fit(X)
print('dbscan2_fit = \n{0}\n'.format(dbscan2_fit))

def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


# plot the result by DBSCAN caluculating
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_dbscan(dbscan, X, size=100)

plt.subplot(122)
plot_dbscan(dbscan2, X, size=600, show_ylabels=False)

save_fig("dbscan_diagram")
plt.show()

dbscan = dbscan2

knn = KNeighborsClassifier(n_neighbors=50, n_jobs=-1)

knn_fit = knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
print('knn_fit = \n{0}\n'.format(knn_fit))

# Check which new data is classified into.
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn_predict_NewData = knn.predict(X_new)
print('knn_predict_NewData = \n{0}\n'.format(knn_predict_NewData))

# Display the judgment probability of individual data.
knn_predict_probability = knn.predict_proba(X_new)
print('knn_predict_probability = \n{0}\n'.format(knn_predict_probability))

# plot the above contents
plt.figure(figsize=(6, 3))
plot_decision_boundaries(knn, X, show_centroids=False)
plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
save_fig("cluster_classification_diagram")
plt.show()

y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
print('y_pred.ravel() = \n{0}\n'.format(y_pred.ravel()))

'''
--------------------------------------------------------------------------------------------------------------------------------
Other Clustering Algorithms
--------------------------------------------------------------------------------------------------------------------------------
'''
print(
    '------------------------------------------------------------------------------------------------------\n'
    '          Spectral Clustering                                                                         \n'
    '------------------------------------------------------------------------------------------------------\n'
)
'''
--------------------------------------------------------------------------------------------------------------------------------
Spectral clustering is a clustering algorithm.

Clustering is classified as unsupervised learning among machine learning methods. 
When data is given, it is a method of dividing data into multiple groups without correct data.

Spectral clustering is characterized by generating a graph from the data and applying the connected component decomposition of the graph to perform clustering. 
Classic clustering algorithms include KMeans and Gaussian mixture models.

KMeans and Gaussian mixture models perform clustering based on the distance from the center of the cluster, 
but spectral clustering focuses on connectivity, so data that could not be clustered well with KMeans and Gaussian mixtures. You may be able to cluster well.
--------------------------------------------------------------------------------------------------------------------------------
'''
sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)

sc1_fit = sc1.fit(X)
print('sc1_fit = \n{0}\n'.format(sc1_fit))

sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)

sc2_fit = sc2.fit(X)
print('sc2_fit = \n{0}\n'.format(sc2_fit))

percentile_95 = np.percentile(sc1.affinity_matrix_, 95)
print('percentile_95 = {0}\n'.format(percentile_95))


def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=size, c='gray', cmap="Paired", alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=10, c=sc.labels_, cmap="Paired")

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)


plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_spectral_clustering(sc1, X, size=500, alpha=0.1)

plt.subplot(122)
plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)

plt.show()

print(
    '------------------------------------------------------------------------------------------------------\n'
    '          Agglomerative Clustering                                                                    \n'
    '------------------------------------------------------------------------------------------------------\n'
)


def learned_parameters(model):
    return [m for m in dir(model)
            if m.endswith("_") and not m.startswith("_")]


# Create datas
X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)

# clustering
agg = AgglomerativeClustering(linkage="complete").fit(X)

print('learned_parameters(agg) = \n{0}\n'.format(learned_parameters(agg)))

print('agg.children_ = \n{0}\n'.format(agg.children_))

print(
    '------------------------------------------------------------------------------------------------------\n'
    '          Gaussian Mixtures                                                                           \n'
    '------------------------------------------------------------------------------------------------------\n'
)
# create 1st datas by make_blobs
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)

# plot scatter
plt.figure(figsize=(8, 6))
plt.title("raw data of make_blobs")
plt.scatter(X1[:, 0][y1 == 0], X1[:, 1][y1 == 0], c='red', label="X10")
plt.scatter(X1[:, 0][y1 == 1], X1[:, 1][y1 == 1], c='blue', label="X11")
plt.grid(True)
plt.legend()
plt.show()

#
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))

plt.figure(figsize=(8, 6))
plt.title(" raw datas after conversion")
plt.scatter(X1[:, 0][y1 == 0], X1[:, 1][y1 == 0], c='red', label="X10")
plt.scatter(X1[:, 0][y1 == 1], X1[:, 1][y1 == 1], c='blue', label="X11")
plt.grid(True)
plt.legend()
plt.show()

# create 2nd datas by make_blobs
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)

for i, y in enumerate(y2):
    if y == 0:
        y2[i] = 2

# plot scatter
plt.figure(figsize=(8, 6))
plt.title("raw data of make_blobs")
plt.scatter(X2[:, 0][y2 == 2], X2[:, 1][y2 == 2], c='green', label="X20")
plt.grid(True)
plt.legend()
plt.show()

X2 = X2 + [6, -8]

plt.figure(figsize=(8, 6))
plt.title(" raw datas after conversion")
plt.scatter(X2[:, 0][y2 == 2], X2[:, 1][y2 == 2], c='green', label="X20")
plt.grid(True)
plt.legend()
plt.show()

# join datas
X = np.r_[X1, X2]
y = np.r_[y1, y2]

#
plt.figure(figsize=(8, 6))
plt.title("integrated raw data")
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c='red', label="X1")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c='blue', label="X2")
plt.scatter(X[:, 0][y == 2], X[:, 1][y == 2], c='green', label="X3")
plt.grid(True)
plt.legend()
plt.show()

'''
--------------------------------------------------------------------------------------------------------------------
Let's train a Gaussian mixture model on the previous dataset:
--------------------------------------------------------------------------------------------------------------------
'''
'''
A Gaussian mixture model is a probabilistic model 
that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. 

One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data 
as well as the centers of the latent Gaussians.
'''
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)

gm_fit = gm.fit(X)
print('gm_fit = \n{0}\n'.format(gm_fit))

'''
-------------------------------------------------------------------------------------------------------------------
Let's look at the parameters that the EM algorithm estimated:
-------------------------------------------------------------------------------------------------------------------
'''
gm_weight = gm.weights_
print('gm_weight = \n{0}\n'.format(gm_weight))

gm_means = gm.means_
print('gm_means = \n{0}\n'.format(gm_means))

gm_covariances = gm.covariances_
print('gm_covariances = \n{0}\n'.format(gm_covariances))

# Did the algorithm actually converge?
print('gm.converged_ = {0}\n'.format(gm.converged_))

# Yes, good. How many iterations did it take?
print('gm.n_iter_ = {0}\n'.format(gm.n_iter_))

'''
---------------------------------------------------------------------------------------------------------------------
You can now use the model to predict which cluster each instance belongs to (hard clustering) 
or the probabilities that it came from each cluster. 
For this, just use predict() method or the predict_proba() method:
---------------------------------------------------------------------------------------------------------------------
'''
gm_predict = gm.predict(X)
print('gm_predict = \n{0}\n'.format(gm_predict))

gm_predict_probablity = gm.predict_proba(X)
print('gm_predict_probablity = \n{0}\n'.format(gm_predict_probablity))

# This is a generative model, so you can sample new instances from it (and get their labels):
X_new, y_new = gm.sample(6)

print('X_new = \n{0}\n'.format(X_new))

print('y_new = \n{0}\n'.format(y_new))

'''
---------------------------------------------------------------------------------------------------------------------
Notice that they are sampled sequentially from each cluster.
---------------------------------------------------------------------------------------------------------------------
'''

# You can also estimate the log of the probability density function (PDF) at any location using the score_samples() method:
gm_score_samples = gm.score_samples(X)
print('gm_score_samples = \n{0}\n'.format(gm_score_samples))

'''
---------------------------------------------------------------------------------------------------------------------
Let's check that the PDF integrates to 1 over the whole space. 
We just take a large square around the clusters, and chop it into a grid of tiny squares, 
then we compute the approximate probability that the instances will be generated in each tiny square
(by multiplying the PDF at one corner of the tiny square by the area of the square), and finally summing all these probabilities). 
The result is very close to 1:
----------------------------------------------------------------------------------------------------------------------
'''
resolution = 100

grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1 / resolution) ** 2
pdf_probas_sum = pdf_probas.sum()
print('pdf_probas_sum = {0}\n'.format(pdf_probas_sum))


# Now let's plot the resulting decision boundaries (dashed lines) and density contours:
def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12), linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, X)

save_fig("gaussian_mixtures_diagram")
plt.show()

'''
---------------------------------------------------------------------------------------------------------------------
You can impose constraints on the covariance matrices that the algorithm looks for by setting the covariance_type hyperparameter:

* "full" (default): no constraint, all clusters can take on any ellipsoidal shape of any size.

* "tied": all clusters must have the same shape, which can be any ellipsoid (i.e., they all share the same covariance matrix).

* "spherical": all clusters must be spherical, but they can have different diameters (i.e., different variances).

* "diag": clusters can take on any ellipsoidal shape of any size, but the ellipsoid's axes must be parallel to the axes 
    (i.e., the covariance matrices must be diagonal).
----------------------------------------------------------------------------------------------------------------------
'''
# option
gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)

# fitting
gm_full_fit = gm_full.fit(X)
print('gm_full_fit = \n{0}\n'.format(gm_full_fit))

gm_tied_fit = gm_tied.fit(X)
print('gm_tied_fit = \n{0}\n'.format(gm_tied_fit))

gm_spherical_fit = gm_spherical.fit(X)
print('gm_spherical_fit = \n{0}\n'.format(gm_spherical_fit))

gm_diag_fit = gm_diag.fit(X)
print('gm_diag_fit = \n{0}\n'.format(gm_diag_fit))


def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)


# compare gm_tired and gm_spherical
compare_gaussian_mixtures(gm_tied, gm_spherical, X)

save_fig("covariance_type_diagram")
plt.show()

# compare gm_full and gm_diag
compare_gaussian_mixtures(gm_full, gm_diag, X)
plt.tight_layout()
plt.show()

print(
    '------------------------------------------------------------------------------------------------------\n'
    '          Anomaly Detection using Gaussian Mixtures                                                   \n'
    '------------------------------------------------------------------------------------------------------\n'
)
'''
---------------------------------------------------------------------------------------------------------------
Gaussian Mixtures can be used for anomaly detection: 
instances located in low-density regions can be considered anomalies. 

You must define what density threshold you want to use. For example, 
in a manufacturing company that tries to detect defective products, the ratio of defective products is usually well-known. 

Say it is equal to 4%, 
then you can set the density threshold to be the value that results in having 4% of the instances located in areas below that threshold density:
---------------------------------------------------------------------------------------------------------------
'''
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

plt.figure(figsize=(8, 4))
plot_gaussian_mixture(gm, X)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)

save_fig("mixture_anomaly_detection_diagram")
plt.show()
print()

print(
    '------------------------------------------------------------------------------------------------------\n'
    '          Model selection                                                                             \n'
    '------------------------------------------------------------------------------------------------------\n'
)
'''
-------------------------------------------------------------------------------------------------------------------
We cannot use the inertia or the silhouette score because they both assume that the clusters are spherical. 
Instead, we can try to find the model that minimizes a theoretical information criterion 
such as the Bayesian Information Criterion (BIC) or the Akaike Information Criterion (AIC):

𝐵𝐼𝐶=log(𝑚)𝑝−2log(𝐿̂ ) 

𝐴𝐼𝐶=2𝑝−2log(𝐿̂ ) 

    * 𝑚  is the number of instances.

    * p is the number of parameters learned by the model.

    * 𝐿̂   is the maximized value of the likelihood function of the model. 
        This is the conditional probability of the observed data  𝐗 , given the model and its optimized parameters.
--------------------------------------------------------------------------------------------------------------------
'''
# Both BIC and AIC penalize models that have more parameters to learn (e.g., more clusters),
# and reward models that fit the data well (i.e., models that give a high likelihood to the observed data).

gm_bic = gm.bic(X)
print('gm_bic = {0}\n'.format(gm_bic))

gm_aic = gm.aic(X)
print('(gm_aic = {0}\n'.format(gm_aic))

# We could compute the BIC manually like this:
n_clusters = 3
n_dims = 2
n_params_for_weights = n_clusters - 1
n_params_for_means = n_clusters * n_dims
n_params_for_covariance = n_clusters * n_dims * (n_dims + 1) // 2
n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance
max_log_likelihood = gm.score(X) * len(X)  # log(L^)
bic = np.log(len(X)) * n_params - 2 * max_log_likelihood
aic = 2 * n_params - 2 * max_log_likelihood

print('manual bic = {0}, aic = {1}\n'.format(bic, aic))

print('manual n_params = {0}\n'.format(n_params))

'''
---------------------------------------------------------------------------------------------------------------------
There's one weight per cluster, 
but the sum must be equal to 1, so we have one degree of freedom less, 
hence the -1. Similarly, the degrees of freedom for an  𝑛×𝑛  covariance matrix is not  𝑛2 , 
but  1+2+⋯+𝑛=𝑛(𝑛+1)/2.
---------------------------------------------------------------------------------------------------------------------
'''
# Let's train Gaussian Mixture models with various values of  𝑘  and measure their BIC:
gms_per_k = [
    GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X) for k in range(1, 11)
]

# caluclate bics and aics
bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]

# Effect of k on aic and bic
plt.figure(figsize=(8, 3))
plt.plot(range(1, 11), bics, "bo-", label="BIC")
plt.plot(range(1, 11), aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
plt.annotate('Minimum', xy=(3, bics[2]), xytext=(0.35, 0.6), textcoords='figure fraction', fontsize=14,
             arrowprops=dict(facecolor='black', shrink=0.1))
plt.legend()
save_fig("aic_bic_vs_k_diagram")
plt.show()
print()

# Let's search for best combination of values for both the number of clusters and the covariance_type hyperparameter:
min_bic = np.infty

for k in range(1, 11):
    for covariance_type in ("full", "tied", "spherical", "diag"):
        bic = GaussianMixture(n_components=k, n_init=10, covariance_type=covariance_type, random_state=42).fit(X).bic(X)
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_covariance_type = covariance_type

print('best_k = {0}\n'.format(best_k))
print('best_covariance_type = {0}\n'.format(best_covariance_type))

print(
    '------------------------------------------------------------------------------------------------------\n'
    '          Variational Bayesian Gaussian Mixtures                                                      \n'
    '------------------------------------------------------------------------------------------------------\n'
)
'''
----------------------------------------------------------------------------------------------------------------
Rather than manually searching for the optimal number of clusters, 
it is possible to use instead the BayesianGaussianMixture class 
which is capable of giving weights equal (or close) to zero to unnecessary clusters. 

Just set the number of components to a value that you believe is greater than the optimal number of clusters, 
and the algorithm will eliminate the unnecessary clusters automatically.
----------------------------------------------------------------------------------------------------------------
'''
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)

bgm_fit = bgm.fit(X)
print('bgm_fit = \n{0}\n'.format(bgm_fit))

# The algorithm automatically detected that only 3 components are needed:
bgm_around = np.round(bgm.weights_, 2)
print('bgm_around = \n{0}\n'.format(bgm_around))

# plot BayesianGaussianMixture
plt.figure(figsize=(8, 5))
plot_gaussian_mixture(bgm, X)
plt.show()
print()

bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1, weight_concentration_prior=0.01,
                                  random_state=42)
bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1, weight_concentration_prior=10000,
                                   random_state=42)

nn = 73
bgm_low_fit = bgm_low.fit(X[:nn])
print('bgm_low_fit = \n{0}\n'.format(bgm_low_fit))

bgm_high_fit = bgm_high.fit(X[:nn])
print('bgm_high_fit = \n{0}\n'.format(bgm_high_fit))

bgm_low_weight = np.round(bgm_low.weights_, 2)
print('bgm_low_weight = \n{0}\n'.format(bgm_low_weight))

bgm_high_weight = np.round(bgm_high.weights_, 2)
print('bgm_high_weight = \n{0}\n'.format(bgm_high_weight))

plt.figure(figsize=(9, 4))

plt.subplot(121)
plot_gaussian_mixture(bgm_low, X[:nn])
plt.title("weight_concentration_prior = 0.01", fontsize=14)

plt.subplot(122)
plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
plt.title("weight_concentration_prior = 10000", fontsize=14)

save_fig("mixture_concentration_prior_diagram")
plt.show()
print()

'''
----------------------------------------------------------------------------------------------------------------------------------
Note: 
the fact that you see only 3 regions in the right plot although there are 4 centroids is not a bug. 

The weight of the top-right cluster is much larger than the weight of the lower-right cluster, 
so the probability that any given point in this region belongs to the top right cluster is greater than 
the probability that it belongs to the lower-right cluster.
----------------------------------------------------------------------------------------------------------------------------------
'''

# in case of make_moons
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)

bgm_fit = bgm.fit(X_moons)
print('bgm_fit = \n{0}\n'.format(bgm_fit))

# Output result
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_data(X_moons)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.subplot(122)
plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)

save_fig("moons_vs_bgm_diagram")
plt.show()
print()

'''
----------------------------------------------------------------------------------------------------------------------------------
Oops, not great... instead of detecting 2 moon-shaped clusters, the algorithm detected 8 ellipsoidal clusters.
However, the density plot does not look too bad, so it might be usable for anomaly detection.
----------------------------------------------------------------------------------------------------------------------------------
'''

print(
    '------------------------------------------------------------------------------------------------------\n'
    '          Likelihood Function                                                                         \n'
    '------------------------------------------------------------------------------------------------------\n'
)
# Prepare data
xx = np.linspace(-6, 4, 101)
ss = np.linspace(1, 2, 101)
XX, SS = np.meshgrid(xx, ss)
ZZ = 2 * norm.pdf(XX - 1.0, 0, SS) + norm.pdf(XX + 4.0, 0, SS)
ZZ = ZZ / ZZ.sum(axis=1) / (xx[1] - xx[0])

plt.figure(figsize=(8, 4.5))

x_idx = 85
s_idx = 30

plt.subplot(221)
plt.contourf(XX, SS, ZZ, cmap="GnBu")
plt.plot([-6, 4], [ss[s_idx], ss[s_idx]], "k-", linewidth=2)
plt.plot([xx[x_idx], xx[x_idx]], [1, 2], "b-", linewidth=2)
plt.xlabel(r"$x$")
plt.ylabel(r"$\theta$", fontsize=14, rotation=0)
plt.title(r"Model $f(x; \theta)$", fontsize=14)

plt.subplot(222)
plt.plot(ss, ZZ[:, x_idx], "b-")
max_idx = np.argmax(ZZ[:, x_idx])
max_val = np.max(ZZ[:, x_idx])
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [0, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.text(1.01, max_val + 0.005, r"$\hat{L}$", fontsize=14)
plt.text(ss[max_idx] + 0.01, 0.055, r"$\hat{\theta}$", fontsize=14)
plt.text(ss[max_idx] + 0.01, max_val - 0.012, r"$Max$", fontsize=12)
plt.axis([1, 2, 0.05, 0.15])
plt.xlabel(r"$\theta$", fontsize=14)
plt.grid(True)
plt.text(1.99, 0.135, r"$=f(x=2.5; \theta)$", fontsize=14, ha="right")
plt.title(r"Likelihood function $\mathcal{L}(\theta|x=2.5)$", fontsize=14)

plt.subplot(223)
plt.plot(xx, ZZ[s_idx], "k-")
plt.axis([-6, 4, 0, 0.25])
plt.xlabel(r"$x$", fontsize=14)
plt.grid(True)
plt.title(r"PDF $f(x; \theta=1.3)$", fontsize=14)
verts = [(xx[41], 0)] + list(zip(xx[41:81], ZZ[s_idx, 41:81])) + [(xx[80], 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
plt.gca().add_patch(poly)

plt.subplot(224)
plt.plot(ss, np.log(ZZ[:, x_idx]), "b-")
max_idx = np.argmax(np.log(ZZ[:, x_idx]))
max_val = np.max(np.log(ZZ[:, x_idx]))
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [-5, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.axis([1, 2, -2.4, -2])
plt.xlabel(r"$\theta$", fontsize=14)
plt.text(ss[max_idx] + 0.01, max_val - 0.05, r"$Max$", fontsize=12)
plt.text(ss[max_idx] + 0.01, -2.39, r"$\hat{\theta}$", fontsize=14)
plt.text(1.01, max_val + 0.02, r"$\log \, \hat{L}$", fontsize=14)
plt.grid(True)
plt.title(r"$\log \, \mathcal{L}(\theta|x=2.5)$", fontsize=14)

save_fig("likelihood_function_diagram")
plt.show()