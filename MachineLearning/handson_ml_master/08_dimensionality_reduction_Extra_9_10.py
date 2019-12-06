'''
-----------------------------------------------------------------------------------------------
Exercise solutions
-----------------------------------------------------------------------------------------------
'''
print(__doc__)

# common liblary
import time
import timeit
import numpy as np
from scipy.stats import norm
import os
import warnings
import urllib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


from sklearn.datasets import fetch_openml

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



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


print   (
        '------------------------------------------------------------------------------------------------------\n'
        '  Exercise solutions: 9                                                                               \n'
        '  Exercise:                                                                                           \n'
        '     Load the MNIST dataset (introduced in chapter 3) and split it into a training set and a test set \n'
        '     (take the first 60,000 instances for training, and the remaining 10,000 for testing).            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# The MNIST dataset was loaded earlier.
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)

# splite data
X_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

X_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]

'''
-----------------------------------------------------------------------------------------------------------------
Exercise: 
Train a Random Forest classifier on the dataset and time how long it takes, 
then evaluate the resulting model on the test set.
-----------------------------------------------------------------------------------------------------------------
'''
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)

t0 = time.time()
rnd_clf_fit = rnd_clf.fit(X_train, y_train)
t1 = time.time()

print("Training took {0:.2f}s".format(t1 - t0))

# calculate accuracy
y_pred = rnd_clf.predict(X_test)

rad_clf_accuracy = accuracy_score(y_test, y_pred)
print('rad_clf_accuracy = {0}\n'.format(rad_clf_accuracy))

# Exercise: Next, use PCA to reduce the dataset's dimensionality, with an explained variance ratio of 95%.
pca = PCA(n_components=0.95)

X_train_reduced = pca.fit_transform(X_train)
print('X_train_reduced = \n{0}\n'.format(X_train_reduced))

# Exercise: Train a new Random Forest classifier on the reduced dataset and see how long it takes. Was training much faster?
rnd_clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
t0 = time.time()
rnd_clf2_fit = rnd_clf2.fit(X_train_reduced, y_train)
t1 = time.time()

print("Training took {0:.2f}s".format(t1 - t0))

'''
---------------------------------------------------------------------------------------------------------------------
Oh no! 
Training is actually more than twice slower now! How can that be? Well, as we saw in this chapter, 
dimensionality reduction does not always lead to faster training time: 
it depends on the dataset, the model and the training algorithm. 

See figure 8-6 (the manifold_decision_boundary_plot* plots above). 
If you try a softmax classifier instead of a random forest classifier, 
you will find that training time is reduced by a factor of 3 when using PCA. 
Actually, we will do this in a second, but first let's check the precision of the new random forest classifier.
--------------------------------------------------------------------------------------------------------------------
'''

# Exercise: Next evaluate the classifier on the test set: how does it compare to the previous classifier?
X_test_reduced = pca.transform(X_test)

y_pred = rnd_clf2.predict(X_test_reduced)
rnd_clf2_accuracy = accuracy_score(y_test, y_pred)
print('accuracy_score = {0}\n'.format(rnd_clf2_accuracy))

'''
----------------------------------------------------------------------------------------------------------------------
It is common for performance to drop slightly when reducing dimensionality, 
because we do lose some useful signal in the process. 
However, the performance drop is rather severe in this case. 
So PCA really did not help: it slowed down training and reduced performance. :(
----------------------------------------------------------------------------------------------------------------------
'''

# Let's see if it helps when using softmax regression:
log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)

t0 = time.time()
log_clf_fit = log_clf.fit(X_train, y_train)
t1 = time.time()

print("Training took {0:.2f}s".format(t1 - t0))

y_pred = log_clf.predict(X_test)
log_clf_accuracy = accuracy_score(y_test, y_pred)
print('log_clf_accuracy = {0}\n'.format(log_clf_accuracy))

'''
------------------------------------------------------------------------------------------------------------------------
Okay, so softmax regression takes much longer to train on this dataset than the random forest classifier, 
plus it performs worse on the test set. 
But that's not what we are interested in right now, we want to see how much PCA can help softmax regression.
------------------------------------------------------------------------------------------------------------------------
'''

# Let's train the softmax regression model using the reduced dataset:
log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)

t0 = time.time()
log_clf2_fit = log_clf2.fit(X_train_reduced, y_train)
t1 = time.time()

print("Training took {0:.2f}s".format(t1 - t0))

'''
-------------------------------------------------------------------------------------------------------------------------
Nice! Reducing dimensionality led to a 4× speedup. :) Let's check the model's accuracy:
-------------------------------------------------------------------------------------------------------------------------
'''
y_pred = log_clf2.predict(X_test_reduced)
log_clf2_accuracy = accuracy_score(y_test, y_pred)
print('log_clf2_accuracy = {0}\n'.format(log_clf2_accuracy))

'''
--------------------------------------------------------------------------------------------------------------------------
A very slight drop in performance, which might be a reasonable price to pay for a 4× speedup, depending on the application.

So there you have it: PCA can give you a formidable speedup... but not always!
--------------------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '  Exercise solutions: 10                                                                              \n'
        '  Exercise:                                                                                           \n'
        '    Use t-SNE to reduce the MNIST dataset down to two dimensions and plot the result using Matplotlib.\n'
        '    You can use a scatterplot using 10 different colors to represent each image of target class.*     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# The MNIST dataset was loaded above.
# Dimensionality reduction on the full 60,000 images takes a very long time, so let's only do this on a random subset of 10,000 images:

np.random.seed(42)

m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]

# Now let's use t-SNE to reduce dimensionality down to 2D so we can plot the dataset:
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

# Now let's use Matplotlib's scatter() function to plot a scatterplot, using a different color for each digit:
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()

'''
------------------------------------------------------------------------------------------------------------------------------------------
Isn't this just beautiful? :) 
This plot tells us which numbers are easily distinguishable from the others (e.g., 0s, 6s, and most 8s are rather well separated clusters), 
and it also tells us which numbers are often hard to distinguish (e.g., 4s and 9s, 5s and 3s, and so on).
-----------------------------------------------------------------------------------------------------------------------------------------
'''

# Let's focus on digits 3 and 5, which seem to overlap a lot.
plt.figure(figsize=(9,9))
cmap = mpl.cm.get_cmap("jet")
for digit in (2, 3, 5):
    plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=[cmap(digit / 9)])
plt.axis('off')
plt.show()

# Let's see if we can produce a nicer image by running t-SNE on these 3 digits:
idx = (y == 2) | (y == 3) | (y == 5)
X_subset = X[idx]
y_subset = y[idx]

tsne_subset = TSNE(n_components=2, random_state=42)
X_subset_reduced = tsne_subset.fit_transform(X_subset)

plt.figure(figsize=(9,9))
for digit in (2, 3, 5):
    plt.scatter(X_subset_reduced[y_subset == digit, 0], X_subset_reduced[y_subset == digit, 1], c=[cmap(digit / 9)])

plt.axis('off')
plt.show()

'''
-------------------------------------------------------------------------------------------------------------------------
Much better, now the clusters have far less overlap. 
But some 3s are all over the place. 
Plus, there are two distinct clusters of 2s, and also two distinct clusters of 5s. 
It would be nice if we could visualize a few digits from each cluster, to understand why this is the case. 
Let's do that now.
-------------------------------------------------------------------------------------------------------------------------
'''

'''
-------------------------------------------------------------------------------------------------------------------------
Exercise: 
Alternatively, you can write colored digits at the location of each instance, 
or even plot scaled-down versions of the digit images themselves 
(if you plot all digits, the visualization will be too cluttered, 
so you should either draw a random sample or plot an instance only if no other instance has already been plotted at a close distance). 

You should get a nice visualization with well-separated clusters of digits.

Let's create a plot_digits() function that will draw a scatterplot (similar to the above scatterplots) plus write colored digits, 
with a minimum distance guaranteed between these digits. 
If the digit images are provided, they are plotted instead. 
This implementation was inspired from one of Scikit-Learn's excellent examples (plot_lle_digits, based on a different digit dataset).
-------------------------------------------------------------------------------------------------------------------------
'''
def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = mpl.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])), color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)

# Let's try it! First let's just write colored digits:
plot_digits(X_reduced, y)
plt.show()

# Well that's okay, but not that beautiful. Let's try with the digit images:
plot_digits(X_reduced, y, images=X, figsize=(35, 25))
plt.show()

plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))
plt.show()

# Exercise: Try using other dimensionality reduction algorithms such as PCA, LLE, or MDS and compare the resulting visualizations.
# Let's start with PCA. We will also time how long it takes:

t0 = time.time()
X_pca_reduced = PCA(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("PCA took {0:.1f}s.".format(t1 - t0))
plot_digits(X_pca_reduced, y)
plt.show()

# Wow, PCA is blazingly fast! But although we do see a few clusters, there's way too much overlap. Let's try LLE:
t0 = time.time()
X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("LLE took {0:.1f}s.".format(t1 - t0))
plot_digits(X_lle_reduced, y)
plt.show()

'''
--------------------------------------------------------------------------------------------------------------------
That took a while, and the result does not look too good. 
Let's see what happens if we apply PCA first, preserving 95% of the variance:
--------------------------------------------------------------------------------------------------------------------
'''

pca_lle = Pipeline  ([
                    ("pca", PCA(n_components=0.95, random_state=42)),
                    ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),
                    ])

t0 = time.time()
X_pca_lle_reduced = pca_lle.fit_transform(X)
t1 = time.time()
print("PCA+LLE took {0:.1f}s.".format(t1 - t0))
plot_digits(X_pca_lle_reduced, y)
plt.show()

'''
----------------------------------------------------------------------------------------------------------------------
The result is more or less the same, but this time it was almost 4× faster.
----------------------------------------------------------------------------------------------------------------------
'''

# Let's try MDS. It's much too long if we run it on 10,000 instances, so let's just try 2,000 for now:
m = 2000
t0 = time.time()
X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X[:m])
t1 = time.time()
print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))
plot_digits(X_mds_reduced, y[:m])
plt.show()

'''
-----------------------------------------------------------------------------------------------------------------------
Meh. This does not look great, all clusters overlap too much. Let's try with PCA first, perhaps it will be faster?
-----------------------------------------------------------------------------------------------------------------------
'''

pca_mds = Pipeline  ([
                    ("pca", PCA(n_components=0.95, random_state=42)),
                    ("mds", MDS(n_components=2, random_state=42)),
                    ])

t0 = time.time()
X_pca_mds_reduced = pca_mds.fit_transform(X[:2000])
t1 = time.time()
print("PCA+MDS took {:.1f}s (on 2,000 MNIST images).".format(t1 - t0))
plot_digits(X_pca_mds_reduced, y[:2000])
plt.show()

'''
-------------------------------------------------------------------------------------------------------------------------
Same result, and no speedup: PCA did not help (or hurt).
-------------------------------------------------------------------------------------------------------------------------
'''
# Let's try LDA:
t0 = time.time()
X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
t1 = time.time()
print("LDA took {:.1f}s.".format(t1 - t0))
plot_digits(X_lda_reduced, y, figsize=(12,12))
plt.show()

'''
--------------------------------------------------------------------------------------------------------------------------
This one is very fast, and it looks nice at first, until you realize that several clusters overlap severely.
--------------------------------------------------------------------------------------------------------------------------
'''

# Well, it's pretty clear that t-SNE won this little competition, wouldn't you agree? We did not time it, so let's do that now:
t0 = time.time()
X_tsne_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_tsne_reduced, y)
plt.show()

'''
---------------------------------------------------------------------------------------------------------------------------
It's twice slower than LLE, but still much faster than MDS, and the result looks great. 
---------------------------------------------------------------------------------------------------------------------------
'''

# Let's see if a bit of PCA can speed it up:
pca_tsne = Pipeline ([
                    ("pca", PCA(n_components=0.95, random_state=42)),
                    ("tsne", TSNE(n_components=2, random_state=42)),
                    ])

t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_tsne_reduced, y)
plt.show()

'''
-----------------------------------------------------------------------------------------------------------------------------
Yes, PCA roughly gave us a 25% speedup, without damaging the result. We have a winner!
-----------------------------------------------------------------------------------------------------------------------------
'''
