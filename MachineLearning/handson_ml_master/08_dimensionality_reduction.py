# Common imports
import numpy as np
import os
import time
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



'''
--------------------------------------------------------------------
Setup
--------------------------------------------------------------------
'''
# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
CHAPTER_ID = "ensembles"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


'''
----------------------------------------------------------------------
Chapter 8: Dimensionality Reduction
----------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          8.2.1  Projection methods                                                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
# Build 3D dataset:
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

# plot 3d data
sns.set_style("darkgrid")

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("X[0]")
ax.set_ylabel("X[1]")
ax.set_zlabel(("X[2]"))
ax.title("3D data scatter")

ax.plot(X[:, 0], X[:, 1], X[:, 2], marker="o", linestyle='None')

plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '          PCA using SVD decomposition                                                                 \n'
      '------------------------------------------------------------------------------------------------------\n')
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

m, n = X.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)