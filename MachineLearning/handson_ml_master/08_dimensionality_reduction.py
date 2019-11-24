# common liblary
import time
import numpy as np
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import seaborn as sns

from sklearn.datasets import make_swiss_roll
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import mean_squared_error

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
----------------------------------------------------------------------
Chapter 8: Dimension deletion
----------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          8.2.1 Projection methods                                                                    \n'
      '------------------------------------------------------------------------------------------------------\n')

# prepare 3d data
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

# plot 3d data
sns.set_style("darkgrid")

fig = plt.figure()
ax = Axes3D(fig)

# attribute
ax.set_title("3D raw data")
ax.set_xlabel("X[0]")
ax.set_ylabel("X[1]")
ax.set_zlabel("X[2]")
ax.grid(True)

ax.plot(X[:, 0], X[:, 1], X[:, 2], marker='o', linestyle='None')

plt.show()

pca = PCA(n_components=2)
pca.fit_transform(X)

X2D = pca.fit_transform(X)

# Recover the 3D points projected on the plane (PCA 2D subspace)
X3D_inv = pca.inverse_transform(X2D)


# Utility class to draw 3D arrows (copied from http://stackoverflow.com/questions/11140163)
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# Express the plane as a function of x and y.
axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

print('C = \n{0}\n'.format(C))
print('R = \n{0}\n'.format(R))

print('z = \n{0}\n'.format(z))

# Plot the 3D dataset, the plane and the projections on that plane.
fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')

X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
np.linalg.norm(C, axis=0)
ax.add_artist(Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.add_artist(Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.plot([0], [0], [0], "k.")

for i in range(m):
    if X[i, 2] > X3D_inv[i, 2]:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
    else:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")

ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
ax.set_title("A 3D dataset that is close to a 2D subspace")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

'''
------------------------------------------------------------------------------------------------------
Note: If you are using Matplotlib 3.0.0, it has a bug and does not
display 3D graphs properly.
See https://github.com/matplotlib/matplotlib/issues/12239
You should upgrade to a later version. If you cannot, then you can
use the following workaround before displaying each 3D graph:
for spine in ax.spines.values():
     spine.set_visible(False)
------------------------------------------------------------------------------------------------------
'''

save_fig("dataset_3d_plot")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

ax.plot(X2D[:, 0], X2D[:, 1], "k+")
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
ax.plot([0], [0], "ko")
ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.set_title("New 2D dataset after projection")
ax.set_xlabel("$z_1$", fontsize=18)
ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
ax.axis([-1.5, 1.3, -1.2, 1.2])
ax.grid(True)
save_fig("dataset_2d_plot")
plt.show()

# Swiss roll:
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# plot swiss roll
axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_title("Swiss roll")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("swiss_roll_plot")
plt.show()

# Projection in swiss roll
plt.figure(figsize=(14, 4))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis(axes[:4])
plt.title("Projection onto a plane that crushes layer differences")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)

plt.subplot(122)
plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis([4, 15, axes[2], axes[3]])
plt.title("Projection onto a plane that leaves the difference in layers clean")
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)

save_fig("squished_swiss_roll_plot")
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          8.2.2 Manifold learning                                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')

axes = [-11.5, 14, -2, 23, -12, 15]

x2s = np.linspace(axes[2], axes[3], 10)
x3s = np.linspace(axes[4], axes[5], 10)
x2, x3 = np.meshgrid(x2s, x3s)

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = X[:, 0] > 5
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot_wireframe(5, x2, x3, alpha=0.5)
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_title("Swiss Roll with decision boundary")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("manifold_decision_boundary_plot1")
plt.show()

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.title("manifold with decision boundary")
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)

save_fig("manifold_decision_boundary_plot2")
plt.show()

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = 2 * (t[:] - 4) > X[:, 1]
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_title("Swiss Roll without decision boundary")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("manifold_decision_boundary_plot3")
plt.show()

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.plot([4, 15], [0, 22], "b-", linewidth=2)
plt.title("manifold without decision boundary")
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)

save_fig("manifold_decision_boundary_plot4")
plt.show()
print()

'''
----------------------------------------------------------------------------------------------------------------
8.3 PCA
----------------------------------------------------------------------------------------------------------------
'''
angle = np.pi / 5
stretch = 5
m = 200

np.random.seed(3)
X = np.random.randn(m, 2) / 10
X = X.dot(np.array([[stretch, 0], [0, 1]]))  # stretch
X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])  # rotate

u1 = np.array([np.cos(angle), np.sin(angle)])
u2 = np.array([np.cos(angle - 2 * np.pi / 6), np.sin(angle - 2 * np.pi / 6)])
u3 = np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])

X_proj1 = X.dot(u1.reshape(-1, 1))
X_proj2 = X.dot(u2.reshape(-1, 1))
X_proj3 = X.dot(u3.reshape(-1, 1))

plt.figure(figsize=(8, 4))
plt.subplot2grid((3, 2), (0, 0), rowspan=3)
plt.plot([-1.4, 1.4], [-1.4 * u1[1] / u1[0], 1.4 * u1[1] / u1[0]], "k-", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4 * u2[1] / u2[0], 1.4 * u2[1] / u2[0]], "k--", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4 * u3[1] / u3[0], 1.4 * u3[1] / u3[0]], "k:", linewidth=2)
plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
plt.axis([-1.4, 1.4, -1.4, 1.4])
plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.title("pca best projection")
plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)

plt.subplot2grid((3, 2), (0, 1))
plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.title("C1")
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3, 2), (1, 1))
plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.title("C2")
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3, 2), (2, 1))
plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.title("C3")
plt.gca().get_yaxis().set_ticks([])
plt.axis([-2, 2, -1, 1])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)

save_fig("pca_best_projection")
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          PCA using SVD decomposition(singular value decomposition)                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
# prepare 3d data
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X_centered = X - X.mean(axis=0)

print('X_centered = \n{0}\n'.format(X_centered))

# plot 3d data
sns.set_style("darkgrid")

fig = plt.figure()
ax = Axes3D(fig)

# attribute
ax.set_title("3D raw data(center = 0)")
ax.set_xlabel("X_centered[0]")
ax.set_ylabel("X_centered[1]")
ax.set_zlabel("X_centered[2]")
ax.grid(True)

ax.plot(X_centered[:, 0], X_centered[:, 1], X_centered[:, 2], marker='o', linestyle='None')

plt.show()

# SVD decomposition(singular value decomposition)
U, s, Vt = np.linalg.svd(X_centered)

# check unitary matrix(rows x rows)
print('U = \n{0}\n'.format(U))

# check Real diagonal matrix(rows x columns) with the singular values
print('s = \n{0}\n'.format(s))

# check Unitary array(columns x columns)
print('vt = \n{0}\n'.format(Vt))

'''
Note: 
    the svd() function returns U, s and Vt, where Vt is equal to  𝐕𝑇 , the transpose of the matrix  𝐕 . 
    Earlier versions of the book mistakenly said that it returned V instead of Vt. Also, 
    Equation 8-1 should actually contain  𝐕  instead of  𝐕𝑇 , like this:
'''

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.2 Main components of PCA                                                                \n'
      '------------------------------------------------------------------------------------------------------\n')

c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

# get row length and column length
m, n = X.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)
print('S[:n, :n] = \n{0}\n'.format(S))

u_Vt = U.dot(S).dot(Vt)
print('u_Vt = \n{0}\n'.format(u_Vt))

# confirm whether X_centerd == U.dot(S).dot(Vt) or not
print('np.allclose(X_centered, U.dot(S).dot(Vt)) = {0}\n'.format(np.allclose(X_centered, U.dot(S).dot(Vt))))

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.3 Projection to lower-order d dimensions                                                \n'
      '------------------------------------------------------------------------------------------------------\n')

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

X2D_using_svd = X2D

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.4 How to use scikit-learn                                                               \n'
      '------------------------------------------------------------------------------------------------------\n')
# With Scikit-Learn, PCA is really trivial. It even takes care of mean centering for you:
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

print('X2D[:5] = \n{0}\n'.format(X2D[:5]))

print('X2D_using_svd[:5] = \n{0}\n'.format(X2D_using_svd[:5]))

'''
--------------------------------------------------------------------------------------------------------------
Notice that running PCA multiple times on slightly different datasets may result in different results. 
In general the only difference is that some axes may be flipped. 
In this example, PCA using Scikit-Learn gives the same projection as the one given by the SVD approach, 
except both axes are flipped:
--------------------------------------------------------------------------------------------------------------
'''
print('np.allclose(X2D, -X2D_using_svd) = {0}\n'.format(np.allclose(X2D, -X2D_using_svd)))

# Recover the 3D points projected on the plane (PCA 2D subspace)
X3D_inv = pca.inverse_transform(X2D)

'''
---------------------------------------------------------------------------------------------------------------
Of course, there was some loss of information during the projection step, 
so the recovered 3D points are not exactly equal to the original 3D points:
---------------------------------------------------------------------------------------------------------------
'''
# confirm whether the original data equal the recovered data or not.
print('np.allclose(X3D_inv, X) = {0}\n'.format(np.allclose(X3D_inv, X)))

# We can compute the reconstruction error:
reconstruction_error = np.mean(np.sum(np.square(X3D_inv - X), axis=1))
print('reconstruction_error = {0}\n'.format(reconstruction_error))

# The inverse transform in the SVD approach looks like this:
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])

'''
-----------------------------------------------------------------------------------------------------------------
The reconstructions from both methods are not identical 
because Scikit-Learn's PCA class automatically takes care of reversing the mean centering, 
but if we subtract the mean, we get the same reconstruction:
-----------------------------------------------------------------------------------------------------------------
'''
# Equivalent processing by sklearn PCA
X3D_inv_using_sklearn = X3D_inv - pca.mean_

# Check if the process by SVD and the equivalent process by sklearn PCA are the same.
print('np.allclose(X3D_inv_using_svd, X3D_inv_using_sklearn) = {0}\n'.format(
    np.allclose(X3D_inv_using_svd, X3D_inv_using_sklearn)))

# The PCA object gives access to the principal components that it computed:
pca_components = pca.components_
print('pca_components = \n{0}\n'.format(pca_components))

# Compare to the first two principal components computed using the SVD method:
print('Vt[:2] = \n{0}\n'.format(Vt[:2]))

'''
Notice how the axes are flipped.
'''

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.5 explained variance ratio                                                              \n'
      '------------------------------------------------------------------------------------------------------\n')

# Now let's look at the explained variance ratio:
variance_ratio = pca.explained_variance_ratio_
print('variance_ratio = {0}\n'.format(variance_ratio))

'''
The first dimension explains 84.2% of the variance, while the second explains 14.6%.
'''

# By projecting down to 2D, we lost about 1.1% of the variance:
loss_variance = 1 - pca.explained_variance_ratio_.sum()
print('loss_variance = {0}\n'.format(loss_variance))

# Here is how to compute the explained variance ratio using the SVD approach (recall that s is the diagonal of the matrix S):
svd_variance_ratio = np.square(s) / np.square(s).sum()
print('svd_variance_ratio = {0}\n'.format(svd_variance_ratio))

'''
Next, let's generate some nice figures! :)
'''

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.6 Choosing the right order                                                              \n'
      '------------------------------------------------------------------------------------------------------\n')
# prepare raw data
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

'''
-----------------------------------------------------------------------------------------------------------------
Without removing the dimensions, 
calculate the PCA and then calculate the required order to maintain 95% of the training set variance.
-----------------------------------------------------------------------------------------------------------------
'''
pca = PCA()
pca_fit = pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
x_95_cumsum = np.argmax(cumsum >= 0.95)
d = x_95_cumsum + 1
y_95_cumsum = cumsum[x_95_cumsum]
print('nesesary dimension(95%) = {0}\n'.format(d))
print('x_95_cumsum = {0}, y_95_cumsum = {1}\n'.format(x_95_cumsum, y_95_cumsum))

line1_x_95 = np.linspace(0, 700, 700)
line1_y = np.linspace(y_95_cumsum, y_95_cumsum, 700)
line2_x = np.linspace(x_95_cumsum, x_95_cumsum, 200)
line2_y_95 = np.linspace(0, y_95_cumsum, 200)

plt.figure(figsize=(8, 6))
plt.title("Relationship between explained variance ratio and dimension")
plt.plot(cumsum)
plt.scatter(x_95_cumsum, y_95_cumsum, marker='o', c='blue')
plt.plot(line1_x_95, line1_y, linestyle='dashed')
plt.plot(line2_x, line2_y_95, linestyle='dashed')
plt.axis([0, 700, 0, 1])
plt.xlabel('Dimensions')
plt.ylabel('explained variance');
plt.show()

# another method
pca = PCA(n_components=0.95)
X_reduced = pca.fit(X_train)

d = pca.n_components_
print('nesesary dimension(95%) = {0}\n'.format(d))

# Sum of factor contributions after dimension reduction
sum_explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
print('sum_explained_variance_ratio = {0}\n'.format(sum_explained_variance_ratio))

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.7 PCA for compression                                                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
# specify dimensions to 154
pca = PCA(n_components=154)

# fitting
X_reduced = pca.fit_transform(X_train)

# calculate reconstruction error
X_recovered = pca.inverse_transform(X_reduced)


# function
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)

save_fig("mnist_compression_plot")
plt.show()

X_reduced_pca = X_reduced

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.8 Incremental PCA                                                                       \n'
      '------------------------------------------------------------------------------------------------------\n')

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="")  # not shown in the book
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)

X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()
plt.show()

X_reduced_inc_pca = X_reduced

# Let's compare the results of transforming MNIST using regular PCA and incremental PCA. First, the means are equal:
print('np.allclose(pca.mean_, inc_pca.mean_) = {0}\n'.format(np.allclose(pca.mean_, inc_pca.mean_)))

'''
But the results are not exactly identical.
Incremental PCA gives a very good approximate solution, but it's not perfect:
'''
print('np.allclose(X_reduced_pca, X_reduced_inc_pca) = {0}\n'.format(np.allclose(X_reduced_pca, X_reduced_inc_pca)))

print('------------------------------------------------------------------------------------------------------\n'
      '          Using memmap()                                                                              \n'
      '------------------------------------------------------------------------------------------------------\n')
# Let's create the memmap() structure and copy the MNIST data into it. This would typically be done by a first program:
filename = "my_mnist.data"
m, n = X_train.shape

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = X_train

'''
Now deleting the memmap() object will trigger its Python finalizer, which ensures that the data is saved to disk.
'''

del X_mm

# Next, another program would load the data and use it for training:
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca_fit = inc_pca.fit(X_mm)
print('inc_pca_fit = \n{0}\n'.format(inc_pca_fit))

print('------------------------------------------------------------------------------------------------------\n'
      '          8.3.9 Randomized PCA                                                                       \n'
      '------------------------------------------------------------------------------------------------------\n')

rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)

print('------------------------------------------------------------------------------------------------------\n'
      '          Time complexity                                                                             \n'
      '------------------------------------------------------------------------------------------------------\n')
# Let's time regular PCA against Incremental PCA and Randomized PCA, for various number of principal components:
for n_components in (2, 10, 154):
    print("n_components =", n_components)
    regular_pca = PCA(n_components=n_components)
    inc_pca = IncrementalPCA(n_components=n_components, batch_size=500)
    rnd_pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")

    for pca in (regular_pca, inc_pca, rnd_pca):
        t1 = time.time()
        pca.fit(X_train)
        t2 = time.time()
        print("    {}: {:.1f} seconds".format(pca.__class__.__name__, t2 - t1))

# Now let's compare PCA and Randomized PCA for datasets of different sizes (number of instances):
times_rpca = []
times_pca = []
sizes = [1000, 10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000, 500000]
for n_samples in sizes:
    X = np.random.randn(n_samples, 5)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components=2)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)

plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_samples")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")
plt.show()

# And now let's compare their performance on datasets of 2,000 instances with various numbers of features:
times_rpca = []
times_pca = []
sizes = [1000, 2000, 3000, 4000, 5000, 6000]
for n_features in sizes:
    X = np.random.randn(2000, n_features)
    pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components=2)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)

plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_features")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          8.4 Kernel PCA                                                                              \n'
      '------------------------------------------------------------------------------------------------------\n')
# make swiss roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# plot swiss roll
axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_title("Swiss roll")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced

    plt.subplot(subplot)
    # plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
    # plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

save_fig("kernel_pca_plot")
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '          8.4.1 Kernel selection and high parameter tuning                                            \n'
      '------------------------------------------------------------------------------------------------------\n')

'''
---------------------------------------------------------------------------------------------------------------
Since kernelPCA (kPCA) is an unsupervised learning algorithm, 
there is no trivial performance index to help you choose the best kernel and high parameter values.
---------------------------------------------------------------------------------------------------------------
'''
clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression(solver="liblinear"))
])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)

grid_search_fit = grid_search.fit(X, y)
print('grid_search_fit = \n{0}\n'.format(grid_search_fit))

# The best kernel and high parameter values are extracted from best_params_.
print('grid_search.best_params_ = \n{0}\n'.format(grid_search.best_params_))

'''
-----------------------------------------------------------------------------------------------------------------
A complete unsupervised learning method that selects kernels and high parameters with the least reconstruction error
-----------------------------------------------------------------------------------------------------------------
'''
# preimage_plot
plt.figure(figsize=(6, 5))

X_inverse = rbf_pca.inverse_transform(X_reduced_rbf)

ax = plt.subplot(111, projection='3d')
ax.view_init(10, -70)
ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")
ax.set_title("pre-image plot")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

save_fig("preimage_plot", tight_layout=False)
plt.show()

# space by reduced dimensions
X_reduced = rbf_pca.fit_transform(X)

plt.figure(figsize=(11, 4))
plt.subplot(132)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")
plt.title("space by reduced dimensions")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()

'''
-------------------------------------------------------------------------------------------------------------------
In order to perform reconstruction, 
a supervised regression model may be trained using the projected instances as a training set 
and the original instances as targets.
-------------------------------------------------------------------------------------------------------------------
'''
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

'''
note:
     By default, fit_inverse_trasform = false, so there is no inverse_transform () method.
    Make sure fit_inverse_trasform = true.
'''
# Calculate the error of the reconstructed pre-image.
print('mean_squared_error(X, X_preimage) = {0}\n'.format(mean_squared_error(X, X_preimage)))

'''
Using grid search and cross-validation, 
kernels and high parameter values that minimize this reconstructed pre-image error are found.
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          8.5 LLE(locally linear embedding)                                                           \n'
      '------------------------------------------------------------------------------------------------------\n')
# make swiss roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

# plot swiss roll
axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_title("Swiss roll")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)

# plot swiss roll
plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)

save_fig("lle_unrolling_plot")
plt.show()

print('------------------------------------------------------------------------------------------------------\n'
      '    8.6 MDS(multidimensional scaling), Isomap and t-SNE(t-distributed stochastic neighbor embedding)  \n'
      '------------------------------------------------------------------------------------------------------\n')

# MDS(multidimensional scaling)
mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(X)

# Isomap
isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X)

# t-SNE(t-distributed stochastic neighbor embedding)
tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)

# LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_mnist = mnist["data"]
y_mnist = mnist["target"]
lda.fit(X_mnist, y_mnist)
X_reduced_lda = lda.transform(X_mnist)

#
titles = ["MDS", "Isomap", "t-SNE"]

plt.figure(figsize=(11, 4))

for subplot, title, X_reduced in zip((131, 132, 133), titles, (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

save_fig("other_dim_reduction_plot")
plt.show()


def learned_parameters(model):
    return [m for m in dir(model)
            if m.endswith("_") and not m.startswith("_")]