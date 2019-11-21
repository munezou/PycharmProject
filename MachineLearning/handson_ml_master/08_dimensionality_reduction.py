# common liblary
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA



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
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
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

print('------------------------------------------------------------------------------------------------------\n'
      '          PCA using SVD decomposition(singular value decomposition)                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
#
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
pca = PCA(n_components = 2)
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

# Recover the 3D points projected on the plane (PCA 2D subspace).