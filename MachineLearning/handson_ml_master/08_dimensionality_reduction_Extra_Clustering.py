# common liblary
import time
import numpy as np
import os
import warnings
import timeit



import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression


from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

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
