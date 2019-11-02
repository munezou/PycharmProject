'''
------------------------------------------------------------------------------------------------------------------------
Setup
------------------------------------------------------------------------------------------------------------------------
'''
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

from six.moves import urllib

proxy_support = urllib.request.ProxyHandler({'https': 'http://proxy.kanto.sony.co.jp:10080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

'''
------------------------------------------------------------------------------------------------------------------------
3.1 MINIST
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '                   3.1 MINIST                                                            \n'
      '-----------------------------------------------------------------------------------------\n')
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
sort_by_target(mnist) # fetch_openml() returns an unsorted dataset

print('mnist["data"] = \n{0}'.format(mnist['data']))
print()
print('mnist["target"] = \n{0}'.format(mnist['target']))
print()

X, y = mnist["data"], mnist["target"]

print('mnist.data.shape = {0}'.format(mnist.data.shape))
print('X.shape          = {0}'.format(X.shape))
print('y.shape          = {0}'.format(y.shape))
print()

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")

save_fig("some_digit_plot")
plt.show()

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()

print('y[36000] = {0}'.format(y[36000]))
print()

print('---< Prepare data for learning and testing. >---')
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

'''
------------------------------------------------------------------------------------------------------------------------
3.2 Binary classifier training
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '                   3.2 Binary classifier training                                        \n'
      '-----------------------------------------------------------------------------------------\n')

print('---< prepare a data of Binary classifier training >---')
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print()

print('---< Stochastic Gradient Descent: SGD >---')
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_5)

print('sgd_clf.predict([some_digit]) = {0}'.format(sgd_clf.predict([some_digit])))
print()

'''
------------------------------------------------------------------------------------------------------------------------
3.3 Performance
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '                   3.3.1 Binary classifier training                                      \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print('scores = {0}'.format(scores))
print()

print('---< implement unique Cross validation >---')
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

print()

print('---< Result of a dam classifier that classifies all images into classes other than 5. >---')
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
scores_base = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print('scores_base = \n{0}'.format(scores_base))
print()

print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　        3.3.2 Confusion matrix          　                            \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_train_5, y_train_pred)
print('confusion_matrix = \n{0}'.format(conf_mat))
print()

y_train_perfect_predictions = y_train_5
conf_mat_cmp = confusion_matrix(y_train_5, y_train_perfect_predictions)
print('confusion_matrix_complete = \n{0}'.format(conf_mat_cmp))
print()

print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　        3.3.3 Precision and recall      　                            \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.metrics import precision_score, recall_score

prec_score = precision_score(y_train_5, y_train_pred)

print('precision_score(y_train_5, y_train_pred) = {0}'.format(prec_score))
print('precision_score = TP / (TP + FP) = {0}'.format(4344 / (4344 + 1307)))
print()

rec_score = recall_score(y_train_5, y_train_pred)

print('recall_score(y_train_5, y_train_pred) = {0}'.format(rec_score))
print('recall_score = TP / (TP + FN) = {0}'.format(4344 / (4344 + 1077)))
print()

print('---< The F value is the harmonic average of precision and recall. >---')
from sklearn.metrics import f1_score
f_value = f1_score(y_train_5, y_train_pred)
print('f_value = {0}'.format(f_value))
print('f_value = TP / (TP + (FP +FN)/2 ) = {0}'.format(4344 / (4344 + ((1077 + 1307)/2))))
print()

print('-----------------------------------------------------------------------------------------\n'
      '                        3.3.4 Tradeoff between precision and recall                      \n'
      '-----------------------------------------------------------------------------------------\n')
y_scores = sgd_clf.decision_function([some_digit])
print('y_scores = {0}'.format(y_scores))

print('---< threshhold = 0(setting value = in sase of predict()) >---')
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print('y_some_digit_pred = {0}'.format(y_some_digit_pred))
print()

threshold = 20000
y_some_digit_pred = (y_scores > threshold)
print('y_some_digit_pred = {0}'.format(y_some_digit_pred))
print()

'''
Use the cross_val_predict () function to compute scores for all instances of the training set, 
this time returning a decision score rather than a prediction.
'''
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

print('y_scores.shape = {0}'.format(y_scores.shape))
print()

print('---< precision_recall_vs_threshold_plot >---')
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
save_fig("precision_recall_vs_threshold_plot")
plt.show()

print('---< If you decide to aim for 90% compatibility >---')
print('(y_train_pred == (y_scores > 0)).all() = {0}'.format((y_train_pred == (y_scores > 0)).all()))
print()

y_train_pred_90 = (y_scores > 70000)

print('---< Calculate the precision and recall of the forecast for 90% precision. >---')
precision_score_90 = precision_score(y_train_5, y_train_pred_90)
print('precision_score_90 = {0}'.format(precision_score_90))
print()
recall_score_90 = recall_score(y_train_5, y_train_pred_90)
print('recall_score_90 = {0}'.format(recall_score_90))
print()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
save_fig("precision_vs_recall_plot")
plt.show()

print()

'''
------------------------------------------------------------------------------------------------------------------------
3.3.5 Receiver Operating Characteristic
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　   3.3.5 Receiver Operating Characteristic                             \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
save_fig("roc_curve_plot")
plt.show()

print()

'''
AUC (area under the curve) can be used to compare classifiers.
A perfect classifier has a POC AUC of 1, while a pure random classifier has a ROC AUC of 0.5.
'''
from sklearn.metrics import roc_auc_score

roc_auc_score_value = roc_auc_score(y_train_5, y_scores)
print('roc_auc_score_value = {0}'.format(roc_auc_score_value))
print()

'''
The ROC curve is very similar to the PR (precision / recall) curve.
As an indication of which to use, use when the positive class is rare or when you are more concerned about false positives than false negatives.
'''

print('---< Train RandomForrestClassifier and compare ROC curve and ROC AUC score with SGDClassifier. >---')

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
save_fig("roc_curve_comparison_plot")
plt.show()

roc_auc_score_forest = roc_auc_score(y_train_5, y_scores_forest)
print('roc_auc_score_forest = {0}'.format(roc_auc_score_forest))
print()

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score_forest = precision_score(y_train_5, y_train_pred_forest)
print('precision_score_forest = {0}'.format(precision_score_forest))
print()

recall_score_forest = recall_score(y_train_5, y_train_pred_forest)
print('recall_score_forest = {0}'.format(recall_score_forest))
print()

'''
------------------------------------------------------------------------------------------------------------------------
3.4 Multi-class classification
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　         3.4 Multi-class classification                                 \n'
      '-----------------------------------------------------------------------------------------\n')
sgd_clf.fit(X_train, y_train)
predict_5 = sgd_clf.predict([some_digit]) # some_digit = X[36000]

print('predict_5 = {0}'.format(predict_5))
print()

some_digit_scores = sgd_clf.decision_function([some_digit])
print('some_digit_scores = \n{0}'.format(some_digit_scores))
print()

some_digit_scores_max = np.argmax(some_digit_scores)
print('some_digit_scores_max = {0}'.format(some_digit_scores_max))
print()
print('sgd_clf.classes_ = \n{0}'.format(sgd_clf.classes_))
print()
print('sgd_clf.classes_[5] = {0}'.format(sgd_clf.classes_[5]))
print()

'''
To force scikit-learn to use OVO or OVA, use OneVsOneClassifier class or OneVsRestClassifier Class.
'''
print('---< How to use OneVsOneClassifier >---')

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=-np.infty, random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf_predict = ovo_clf.predict([some_digit]) # some_digit = X[36000]
print('ovo_clf_predict = {0}'.format(ovo_clf_predict))
print()

print('ovo_clf.estimators_ = {0}'.format(ovo_clf.estimators_))
print()

print('---< Classifier uses  RandomForrestClassifier. >---')
forest_clf.fit(X_train, y_train)
forest_predict_result = forest_clf.predict([some_digit])
print('forest_predict_result = {0}'.format(forest_predict_result))
print()
forest_predict_probabilistic = forest_clf.predict_proba([some_digit])
print('forest_predict_probabilistic = \n{0}'.format(forest_predict_probabilistic))
print()

print('---< Cross-validate to find the SGDClassifier precision. >---')
SGD_cross_val_score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print('SGD_cross_val_score = \n{0}'.format(SGD_cross_val_score))
print()

print('---< Scale X_train to increase precision. >---')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
SGD_cross_val_score_modify = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print('SGD_cross_val_score_modify = \n{0}'.format(SGD_cross_val_score_modify))
print()

'''
------------------------------------------------------------------------------------------------------------------------
3.5 Analysis of misclassification
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　     3.5 Analysis of misclassification                                 \n'
      '-----------------------------------------------------------------------------------------\n')
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print('conf_mx = \n{0}'.format(conf_mx))
print()

def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

plt.matshow(conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_plot", tight_layout=False)
plt.show()
print()

print('---< Set the diagonal to 0, leave only misclassifications, and plot the results. >---')
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_errors_plot", tight_layout=False)
plt.show()
print()

print('---< error_analysis_digits_plot >---')

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
save_fig("error_analysis_digits_plot")
plt.show()
print()

'''
------------------------------------------------------------------------------------------------------------------------
3.6 Multilabel classification
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　       3.6 Multilabel classification                                   \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf_predict_result = knn_clf.predict([some_digit])  # some_digit = X[36000]
print('knn_clf_predict_result = {0}'.format(knn_clf_predict_result))
print()

print('---< perform Cross-validation predict >---')
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
f1_score_restult = f1_score(y_multilabel, y_train_knn_pred, average="macro")
print('f1_score_restult = {0}'.format(f1_score_restult))
print()

'''
------------------------------------------------------------------------------------------------------------------------
3.7 Multioutput classification
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　       3.7 Multioutput classification                                  \n'
      '-----------------------------------------------------------------------------------------\n')
print("---< Numpy's randint () function adds noise to the MINIST image and creates a training set and a test set. >---")
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
save_fig("noisy_digit_example_plot")
plt.show()

print('---< Remove noise from MINIST images with noise. >---')
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
save_fig("cleaned_digit_example_plot")
plt.show()
print()

'''
------------------------------------------------------------------------------------------------------------------------
Extra material
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　      Dummy (ie. random) classifier                                    \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.dummy import DummyClassifier
dmy_clf = DummyClassifier()
y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_dmy = y_probas_dmy[:, 1]

fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
plot_roc_curve(fprr, tprr)
plt.show()
print()

print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　               KNN classifier                                          \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)

y_knn_pred = knn_clf.predict(X_test)

from sklearn.metrics import accuracy_score
knn_accuracy_score = accuracy_score(y_test, y_knn_pred)
print('knn_accuracy_score = {0}'.format(knn_accuracy_score))
print()

from scipy.ndimage.interpolation import shift
def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)

plot_digit(shift_digit(some_digit, 5, 1, new=100))


X_train_expanded = [X_train]
y_train_expanded = [y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
    X_train_expanded.append(shifted_images)
    y_train_expanded.append(y_train)

X_train_expanded = np.concatenate(X_train_expanded)
y_train_expanded = np.concatenate(y_train_expanded)
X_train_expanded.shape, y_train_expanded.shape
print('(X_train_expanded.shape = {0}, y_train_expanded.shape = {1})'.format(X_train_expanded.shape, y_train_expanded.shape))
print()

knn_clf.fit(X_train_expanded, y_train_expanded)

y_knn_expanded_pred = knn_clf.predict(X_test)

knn_accuracy_score = accuracy_score(y_test, y_knn_expanded_pred)

print('knn_accuracy_score = {0}'.format(knn_accuracy_score))
print()

print('---<  >---')
ambiguous_digit = X_test[2589]
knn_predict_proba = knn_clf.predict_proba([ambiguous_digit])
print('knn_predict_proba = \n{0}'.format(knn_predict_proba))
print()

plot_digit(ambiguous_digit)
plt.show()
print()

'''
------------------------------------------------------------------------------------------------------------------------
Exercise solutions
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　　　　   1. An MNIST Classifier With Over 97% Accuracy                       \n'
      '-----------------------------------------------------------------------------------------\n')
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

knn_best_params_ = grid_search.best_params_
print('knn_best_params_ = {0}'.format(knn_best_params_))
print()

knn_best_score_ = grid_search.best_score_
print('knn_best_score_ = {0}'.format(knn_best_score_))
print()

from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)

knn_accuracy_score = accuracy_score(y_test, y_pred)

print('knn_accuracy_score = {0}'.format(knn_accuracy_score))
print()

print('-----------------------------------------------------------------------------------------\n'
      '           　　                　　　   2. Data Augmentation                                \n'
      '-----------------------------------------------------------------------------------------\n')
from scipy.ndimage.interpolation import shift
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

image = X_train[1000]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()
print()

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier(**grid_search.best_params_)

knn_clf.fit(X_train_augmented, y_train_augmented)

y_pred = knn_clf.predict(X_test)
print('accuracy_score(y_test, y_pred) = {0}'.format(accuracy_score(y_test, y_pred)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
3. Tackle the Titanic dataset
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　               3. Tackle the Titanic dataset                                \n'
      '-----------------------------------------------------------------------------------------\n')
import os

TITANIC_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", "titanic")

import pandas as pd

pd.set_option('display.max_columns', None)

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)

train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")
y_test = load_titanic_data("gender_submission.csv")

'''
The attributes have the following meaning:

*Survived: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
*Pclass: passenger class.
*Name, Sex, Age: self-explanatory
*SibSp: how many siblings & spouses of the passenger aboard the Titanic.
*Parch: how many children & parents of the passenger aboard the Titanic.
*Ticket: ticket id
*Fare: price paid (in pounds)
*Cabin: passenger's cabin number
*Embarked: where the passenger embarked the Titanic
'''
train_data_columns_original = train_data.columns
print('train_data_columns_original = \n{0}'.format(train_data_columns_original))
print('train_data.head() = \n{0}'.format(train_data.head()))
print('train_data.tail() = \n{0}'.format(train_data.tail()))
print()

train_data_info = train_data.info()
print('train_data_info = \n{0}'.format(train_data_info))
print()

train_data_describe = train_data.describe()
print('train_data_describe = \n{0}'.format(train_data_describe))
print()

train_data_Survived_value = train_data["Survived"].value_counts()
print('train_data_Survived_value = \n{0}'.format(train_data_Survived_value))
print()

train_data_Pclass_value = train_data["Pclass"].value_counts()
print('train_data_Pclass_value = \n{0}'.format(train_data_Pclass_value))
print()

train_data_sex_value = train_data["Sex"].value_counts()
print('train_data_sex_value = \n{0}'.format(train_data_sex_value))
print()

'''
The Embarked attribute tells us where the passenger embarked: C=Cherbourg, Q=Queenstown, S=Southampton.
'''
train_data_embarked_value = train_data["Embarked"].value_counts()

print('---< we built in the previous chapter to select specific attributes from the DataFrame. >---')
from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

print('---< numerical pipeline >---')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])

num_pipeline_fit_transform_result = num_pipeline.fit_transform(train_data)
print('num_pipeline_fit_transform_result = \n{0}'.format(num_pipeline_fit_transform_result))
print()

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

print('---< categorial pipeline >---')

from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

cat_pipeline_fit_transfer_result = cat_pipeline.fit_transform(train_data)
print('cat_pipeline_fit_transfer_result = \n{0}'.format(cat_pipeline_fit_transfer_result))
print()

print('---< Finally, let us join the numerical and categorical pipelines. >--- ')
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

print('---< \n'
      'Now we have a nice preprocessing pipeline that takes the raw data and outputs numerical input features that we can feed to any Machine Learning model we want.\n'
      '>---\n')

X_train = preprocess_pipeline.fit_transform(train_data)
X_train_pd = pd.DataFrame(X_train)

y_train = train_data["Survived"]
print('X_train.head() = \n{0}'.format(X_train_pd.head()))
print('y_train.head() = \n{0}'.format(y_train.head()))
print('X_train.tail() = \n{0}'.format(X_train_pd.tail()))
print('y_train.tail() = \n{0}'.format(y_train.tail()))
print()

print('---< We are now ready to train a classifier. Let us start with an SVC >---')
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)

print('---< our model is trained, let us use it to make predictions on the test set >---')
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)
y_pred_pd = pd.DataFrame(y_pred)
y_test_Survived = y_test['Survived']

print('y_pred.head() = \n{0}'.format(y_pred_pd.head()))
print('y_test_Survived.head() = \n{0}'.format(y_test_Survived.head()))
print()
print('y_pred.tail() = \n{0}'.format(y_pred_pd.tail()))
print('y_test_Survived.tail() = \n{0}'.format(y_test_Survived.tail()))
print()

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores_mean = svm_scores.mean()
print('svm_scores_mean = {0}'.format(svm_scores_mean))
print()

print('---< Let us try a RandomForestClassifier >---')

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores_mean = forest_scores.mean()
print('forest_scores_mean = {0}'.format(forest_scores_mean))
print()

'''
Instead of just looking at the mean accuracy across the 10 cross-validation folds, let's plot all 10 scores for each model,
 along with a box plot highlighting the lower and upper quartiles, and "whiskers" showing the extent of the scores. 
(thanks to Nevin Yilmaz for suggesting this visualization)
Note that the boxplot() function detects outliers (called "fliers") and does not include them within the whiskers. 
Specifically, if the lower quartile is  1  and the upper quartile is  3 , 
then the interquartile range  =31  (this is the box's height), and any score lower than  11.5×  is a flier,
 and so is any score greater than  3+1.5× .
'''
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()
print()

train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data_AgeBucket_Suvived = train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
print('train_data_AgeBucket_Suvived = \n{0}'.format(train_data_AgeBucket_Suvived))
print()

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data_RelativesOnboard_Survived = train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
print('train_data_RelativesOnboard_Survived = \n{0}'.format(train_data_RelativesOnboard_Survived))
print()

'''
------------------------------------------------------------------------------------------------------------------------
4. Spam classifier
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------\n'
      '           　　                    4. Spam classifier                                     \n'
      '-----------------------------------------------------------------------------------------\n')
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", "spam")

def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()

fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

print('len(ham_filenames) = {0}'.format(len(ham_filenames)))
print()
print('len(spam_filenames) = {0}'.format(len(spam_filenames)))
print()

'''
We can use Python's email module to parse these emails (this handles headers, encoding, and so on)
'''
import email
#import email.policy
from email.parser import BytesParser, Parser
from email.policy import default

def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

print()

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

print(ham_emails[1].get_content().strip())

print(spam_emails[6].get_content().strip())

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

structures_counter(ham_emails).most_common()

structures_counter(spam_emails).most_common()
print()

print('---< Now let us take a look at the email headers: >---')
for header, value in spam_emails[0].items():
    print(header,":",value)

spam_email0_subject = spam_emails[0]["Subject"]
print('spam_email0_subject = \n{0}'.format(spam_email0_subject))
print()

print('---< Okay, before we learn too much about the data, let us not forget to split it into a training set and a test set: >---')
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

print('---< Let us see if it works. This is HTML spam: >---')
html_spam_emails = [email for email in X_train[y_train==1] if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")
print()

print('---< And this is the resulting plain text: >---')
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")
print()

print('---< \n'
      'Now let us write a function that takes an email as input and returns its content as plain text, whatever its format is: \n'
      '>---\n')

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

print(email_to_text(sample_html_spam)[:100], "...")
print()

'''
Let's throw in some stemming!
 For this to work, you need to install the Natural Language Toolkit (NLTK). 
It's as simple as running the following command.
 (don't forget to activate your virtualenv first; if you don't have one, you will likely need administrator rights, or use the --user option):
'''
import nltk

stemmer = nltk.PorterStemmer()
for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
    print(word, "=>", stemmer.stem(word))

try:
    import urlextract  # may require an Internet connection to download root domain names

    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None

from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

print('---< Let us try this transformer on a few emails: >---')
X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
print('X_few_wordcounts = \n{0}'.format(X_few_wordcounts))
print()

'''
Now we have the word counts, and we need to convert them to vectors. 
For this, we will build another transformer whose fit() method will build the vocabulary (an ordered list of the most common words) 
and whose transform() method will use the vocabulary to convert word counts to vectors. 
The output is a sparse matrix.
'''

from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
print('X_few_vectors = \n{0}'.format(X_few_vectors))
print()

print('X_few_vectors.toarray() = \n{0}'.format(X_few_vectors.toarray()))
print()

'''
What does this matrix mean?
 Well, the 64 in the third row, first column, means that the third email contains 64 words that are not part of the vocabulary. 
The 1 next to it means that the first word in the vocabulary is present once in this email. 
The 2 next to it means that the second word is present twice, and so on. 
You can look at the vocabulary to know which words we are talking about. 
The first word is "of", the second word is "and", etc.
'''

print('vocab_transformer.vocabulary_ = {0}'.format(vocab_transformer.vocabulary_))
print()

print('---< We are now ready to train our first spam classifier! Let us transform the whole dataset: >---')
from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver="liblinear", random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
print('score.mean() = {0}'.format(score.mean()))
print()

from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="liblinear", random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
