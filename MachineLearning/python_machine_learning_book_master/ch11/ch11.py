# common library
import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

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
    path = os.path.join(PROJECT_ROOT_DIR, "figures", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")

'''
--------------------------------------------------------------------------------------------------
Chapter 11 - Working with Unlabeled Data – Clustering Analysis
--------------------------------------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          11.1 Grouping objects by similarity using k-means                                           \n'
      '------------------------------------------------------------------------------------------------------\n')
# create raw data by make_blobs
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

# plot raw data
plt.figure(figsize=(8, 6))
plt.title("raw data by make_blobs")
plt.scatter(X[:, 0], X[:, 1], c='red', marker='o')
plt.grid(True)
save_fig('speres', tight_layout=False)
plt.show()

km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', label='cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='v', label='cluster 3')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', label='centroids')

plt.title("Clustering with using kMeans")
plt.legend()
plt.grid(True)
save_fig('centroids', tight_layout=False)
plt.show()


print('------------------------------------------------------------------------------------------------------\n'
      '          11.1.1 K-means++ method                                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')
# Hard versus soft clustering
print()

print('------------------------------------------------------------------------------------------------------\n'
      '          11.1.3 Using the elbow method to find the optimal number of clusters                        \n'
      '------------------------------------------------------------------------------------------------------\n')
print('Distortion: {0:.2f}\n'.format(km.inertia_))

distortions = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.title('elbo method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
save_fig('elbow', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          11.1.4 Quantifying the quality of clustering via silhouette plots                           \n'
      '------------------------------------------------------------------------------------------------------\n')

km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.title('silhouette')
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

save_fig('silhouette', tight_layout=False)
plt.show()
print()

print('---< Comparison to "bad" clustering: >---')
km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', label='cluster 2')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', label='centroids')
plt.title("bad clustering")
plt.legend()
plt.grid()
save_fig('centroids_bad', tight_layout=False)
plt.show()

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.title('silhouette_bad')
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

save_fig('silhouette_bad', tight_layout=False)
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          11.2 Organizing clusters as a hierarchical tree                                             \n'
      '------------------------------------------------------------------------------------------------------\n')
im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "11_05.png"))
im.show()

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

print('df = \n{0}\n'.format(df))

# plot 3d data
sns.set_style("darkgrid")

fig = plt.figure()
ax = Axes3D(fig)

# attribute
ax.set_title("3D raw data")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)

ax.plot(X[:, 0], X[:, 1], X[:, 2], marker='o', linestyle='None')

plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          11.2.1 Performing hierarchical clustering on a distance matrix                              \n'
      '------------------------------------------------------------------------------------------------------\n')
# Calculate the distance with pdist and create a symmetric matrix with squareform.
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)

print('row_dist = \n{0}\n'.format(row_dist))

'''
---------------------------------------------------------------------------------------------------------------
We can either pass a condensed distance matrix (upper triangular) from the pdist function, 
or we can pass the "original" data array and define the metric='euclidean' argument in linkage. 
However, we should not pass the squareform distance matrix, which would yield different distance values although 
the overall clustering could be the same.
---------------------------------------------------------------------------------------------------------------
'''
def incorrect_approach(row_clusters):
    result = pd.DataFrame(
                        row_clusters,
                        columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                        index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]
                        )
    return result

# 1. incorrect approach: Squareform distance matrix
row_1 = linkage(row_dist, method='complete', metric='euclidean')
incorrect_approach_1 = incorrect_approach(row_clusters=row_1)
print('incorrect_approach_1 = \n{0}\n'.format(incorrect_approach_1))

# 2. correct approach: Condensed distance matrix
row_2 = linkage(pdist(df, metric='euclidean'), method='complete')
incorrect_approach_2 = incorrect_approach(row_clusters=row_2)
print('incorrect_approach_2 = \n{0}\n'.format(incorrect_approach_2))

# 3. correct approach: Input sample matrix
row_3 = linkage(df.values, method='complete', metric='euclidean')
incorrect_approach_3 = incorrect_approach(row_clusters=row_3)
print('incorrect_approach_3 = \n{0}\n'.format(incorrect_approach_3))

# Display the calculation results in a tree diagram.
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])

row_dendr = dendrogram  (
                        row_3, 
                        labels=labels,
                        # make dendrogram black (part 2/2)
                        # color_threshold=np.inf
                        )
plt.tight_layout()
plt.ylabel('Euclidean distance')
save_fig('dendrogram', tight_layout=False)
plt.show()
print()

print  ('------------------------------------------------------------------------------------------------------\n'
        '          11.2.2 Attaching dendrograms to a heat map                                                  \n'
        '------------------------------------------------------------------------------------------------------\n')
# plot row dendrogram
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])

# note: for matplotlib < v1.5.1, please use orientation='right'
row_dendr = dendrogram(row_3, orientation='left')

# reorder data with respect to clustering
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

axd.set_xticks([])
axd.set_yticks([])

# remove axes spines from dendrogram
for i in axd.spines.values():
        i.set_visible(False)

# plot heatmap
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-pos, y-pos, width, height
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))

save_fig('heatmap', tight_layout=False)
plt.show()

print  ('------------------------------------------------------------------------------------------------------\n'
        '          11.2.3 Applying agglomerative clustering via scikit-learn                                   \n'
        '------------------------------------------------------------------------------------------------------\n')
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)

print  ('------------------------------------------------------------------------------------------------------\n'
        '          11.3 Locating regions of high density via DBSCAN                                            \n'
        '------------------------------------------------------------------------------------------------------\n')
im = Image.open(os.path.join(PROJECT_ROOT_DIR,"Images", "11_11.png"))
im.show()

# create make_moon
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
save_fig('moons', tight_layout=False)
plt.show()

# K-means and hierarchical clustering:
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# create data with KMeabs
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)

# disply data by KMeans
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')

# create data with AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(X)

# display dagta by AgglomerativeClustering
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red', marker='s', s=40, label='cluster 2')
ax2.set_title('Agglomerative clustering')

plt.legend()
save_fig('kmeans_and_ac', tight_layout=False)
plt.show()
print()

# Density-based clustering:
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)

plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c='red', marker='s', s=40, label='cluster 2')

plt.legend()
save_fig('moons_dbscan', tight_layout=False)
plt.show()
