# Common imports
import numpy as np
import os
import time
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_moons
from sklearn.datasets import make_regression
import xgboost

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


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
Chapter 7: Ensemble learning and random forest
----------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          7.1 Voting classifier                                                                       \n'
      '------------------------------------------------------------------------------------------------------\n')
heads_proba = 0.51
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8,3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
save_fig("law_of_large_numbers_plot")
plt.show()

'''
-------------------------------------------------------------------------------
Create and train a voting classifier composed of three different classifiers.
-------------------------------------------------------------------------------
'''
# create moons data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# display raw data of moons.
plt.figure(figsize=(8, 6))
plt.title("raw data of moons(noise=0.3, random_state=42)")
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c='red', label="y = 0")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c='blue', label="y = 1")
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.grid(True)
plt.legend()
plt.show()

print()

# split  data to test data and train data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print('---< setting classifier(voting="hard") >---')
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

# fitting
volt_clf_fit = voting_clf.fit(X_train, y_train)
print('volt_clf_fit = \n{0}\n'.format(volt_clf_fit))

# Performance evaluation
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

print()

print('---< setting classifier(voting="soft") >---')
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')

# fitting
volt_clf_fit = voting_clf.fit(X_train, y_train)
print('volt_clf_fit = \n{0}\n'.format(volt_clf_fit))

# Performance evaluation
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

print()

print('------------------------------------------------------------------------------------------------------\n'
      '          7.2.1 Bagging and pasting in scikit-learn                                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
print('---< prepare a Bagging Classifier >---')
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)

# fitting
bag_clf_fit = bag_clf.fit(X_train, y_train)
print('bag_clf_fit = \n{0}\n'.format(bag_clf_fit))

# accuracy result
y_pred = bag_clf.predict(X_test)

# compare target and predict
print('accuracy_score(y_test, y_pred) = {0}\n'.format(accuracy_score(y_test, y_pred)))

print('---< prepare a Decision trees >---')
tree_clf = DecisionTreeClassifier(random_state=42)

# fitting
tree_clf.fit(X_train, y_train)

# accuracy result
y_pred_tree = tree_clf.predict(X_test)

# compare target and predict
print('accuracy_score(y_test, y_pred_tree) = {0}\n'.format(accuracy_score(y_test, y_pred_tree)))

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()

print()

print('------------------------------------------------------------------------------------------------------\n'
      '          7.2.2 Out-of-Bag evaluation                                                                 \n'
      '------------------------------------------------------------------------------------------------------\n')
# Bagging Classifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, bootstrap_features=True, n_jobs=-1, oob_score=True)

# fitting
bag_clf_fit = bag_clf.fit(X_train, y_train)
print('bag_clf_fit = \n{0}\n'.format(bag_clf_fit))

# OOB verification
bag_clf_score = bag_clf.oob_score_
print('bag_clf_score = {0}\n'.format(bag_clf_score))

# accuracy score
y_pred = bag_clf.predict(X_test)
bad_clf_accuracy = accuracy_score(y_test, y_pred)
print('bad_clf_accuracy = {0}\n'.format(bad_clf_accuracy))

# oob_decision function by X_train
bad_clf_oob_decision = bag_clf.oob_decision_function_
print('bad_clf_oob_decision = \n{0}\n'.format(bad_clf_oob_decision))

print('------------------------------------------------------------------------------------------------------\n'
      '          7.4 Random Forests                                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# Bagging Classifier with Decision Trees
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)

# fiting and predict
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)

# fiting and predict
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

# different of Bagging Classifier with Decision and Random Forest Classifier
diff = np.sum(y_pred == y_pred_rf) / len(y_pred)
print('diff = {0}\n'.format(diff))

print('------------------------------------------------------------------------------------------------------\n'
      '          7.4.2 Importance of features                                                                \n'
      '------------------------------------------------------------------------------------------------------\n')

# load iris data
iris = load_iris()

# Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

# fitting
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

print()

rnd_clf_feature_importance = rnd_clf.feature_importances_
print('rnd_clf_feature_importance = \n{0}\n'.format(rnd_clf_feature_importance))

plt.figure(figsize=(6, 4))

for i in range(15):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.02, contour=False)

plt.show()
print()

# load mnist data
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)

# Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)

# fitting
rnd_clf_fit = rnd_clf.fit(mnist["data"], mnist["target"])

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.hot, interpolation="nearest")
    plt.axis("off")

plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])

save_fig("mnist_feature_importance_plot")
plt.show()

'''
-----------------------------------------------------------------------------------
7.5 Boosting
-----------------------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          7.5.1 AdaBoost                                                                              \n'
      '------------------------------------------------------------------------------------------------------\n')
# create moons data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# split  data to test data and train data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# AdaBoost Classifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)

# fitting
ada_clf_fit = ada_clf.fit(X_train, y_train)

plot_decision_boundary(ada_clf, X, y)

m = len(X_train)

plt.figure(figsize=(11, 4))
for subplot, learning_rate in ((121, 1), (122, 0.5)):
    sample_weights = np.ones(m)
    plt.subplot(subplot)
    for i in range(5):
        svm_clf = SVC(kernel="rbf", C=0.05, gamma="auto", random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = svm_clf.predict(X_train)
        sample_weights[y_pred != y_train] *= (1 + learning_rate)
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
    if subplot == 121:
        plt.text(-0.7, -0.65, "1", fontsize=14)
        plt.text(-0.6, -0.10, "2", fontsize=14)
        plt.text(-0.5,  0.10, "3", fontsize=14)
        plt.text(-0.4,  0.55, "4", fontsize=14)
        plt.text(-0.3,  0.90, "5", fontsize=14)

save_fig("boosting_plot")
plt.show()
print()

list_ada = list(m for m in dir(ada_clf) if not m.startswith("_") and m.endswith("_"))
print('list_ada = \n{0}\n'.format(list_ada))

print('------------------------------------------------------------------------------------------------------\n'
      '          7.5.2 Gradient Boosting                                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')
# raw data
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

# plot raw data
plt.figure(figsize=(8, 6))
plt.title("Raw data used for Gradient Boosting")
plt.scatter(X, y, c='pink', label="raw data")
plt.grid(True)
plt.show()

# Regressor is Desion Trees
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)

# fitting
tree_reg1_fit = tree_reg1.fit(X, y)
print('tree_reg1_fit = \n{0}\n'.format(tree_reg1_fit))

# Fit the new predictor to the residual of the previous predictor. (First time)
y2 = y - tree_reg1.predict(X)

# new regressor is Desion Trees
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)

# fitting to new regressor
tree_reg2_fit = tree_reg2.fit(X, y2)
print('tree_reg2_fit = \n{0}\n'.format(tree_reg2_fit))

# Fit the new predictor to the residual of the previous predictor. (2nd time)
y3 = y2 - tree_reg2.predict(X)

# new regressor is Desion Trees
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)

# fitting to new regressor
tree_reg3_fit = tree_reg3.fit(X, y3)
print('tree_reg3_fit = \n{0}\n'.format(tree_reg3_fit))

# Calculate the expected value (y) when X = 0.8.
X_new = np.array([[0.8]])

y_pred =sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

print('y_predict = {0}, when X = 0.8.\n'.format(y_pred))

# The left plot shows the prediction of three decision trees, and the right plot shows the prediction of the ensemble.
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

save_fig("gradient_boosting_plot")
plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '   By using sklearn GradientBoostingRegressor for the above ensemble prediction,                      \n'
      '   GBRT ensemble can be trained more easily.                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# Regressor is Gradient Bossting(n_estimators=3, learning_rate=1.0)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)

# fitting
gbrt_fit = gbrt.fit(X, y)
print('gbrt_fit = \n{0}\n'.format(gbrt_fit))

# Regressor is Gradient Bossting(n_estimators=200, learning_rate=0.1)
gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)

# fitting
gbrt_slow_fit = gbrt_slow.fit(X, y)
print('gbrt_slow_fit = \n{0}\n'.format(gbrt_slow_fit))

# Display result
plt.figure(figsize=(11,4))

plt.subplot(121)
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)

save_fig("gbrt_learning_rate_plot")
plt.show()
print()

print('------------------------------------------------------------------------------------------------------\n'
      '   Gradient Boosting with Early stopping                                                              \n'
      '------------------------------------------------------------------------------------------------------\n')
# prepare test data and train data.
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

# Regressor is GradientBoostingRegressor(warm_stat=False)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)

# fitting
gbrt_fit = gbrt.fit(X_train, y_train)

# Calculate an error between real value and predict value.
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]

bst_n_estimators = np.argmin(errors) + 1

# Regressor is GradientBoostingRegressor
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)

# fitting
gbrt_best_fit = gbrt_best.fit(X_train, y_train)
print('gbrt_best_fit = \n{0}\n'.format(gbrt_best_fit))

# Extract minimum value of errors
min_error = np.min(errors)

#
plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)

save_fig("early_stopping_gbrt_plot")
plt.show()

# Regressor is GradientBoostingRegressor(warm_stat=True)
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

'''
-------------------------------------------------------------------------------------
 Set warm_start = True to end training early.
 When the method is called, 
it leaves the existing decision tree so that it can be trained progressively.
 Next, when the verification error is not repeated 5 times, the training is terminated.
-------------------------------------------------------------------------------------
'''

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping

print('gbrt.n_estimators = {0}\n'.format(gbrt.n_estimators))

print("Minimum validation MSE:", min_val_error)

print('------------------------------------------------------------------------------------------------------\n'
      '          Using XGBoost                                                                               \n'
      '------------------------------------------------------------------------------------------------------\n')
if xgboost is not None:  # not shown in the book
    xgb_reg = xgboost.XGBRegressor(random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
print("Validation MSE: {0}\n".format(val_error))

if xgboost is not None:  # not shown in the book
    xgb_reg.fit(X_train, y_train,
                eval_set=[(X_val, y_val)], early_stopping_rounds=2)
    y_pred = xgb_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    print("Validation MSE: {0}\n".format(val_error))

start = time.time()
xgboost.XGBRegressor().fit(X_train, y_train) if xgboost is not None else None
elapsed_time = time.time() - start
print ("XGBRegressor_elapsed_time:{0} [sec]\n".format(elapsed_time))

#print(timeit.timeit('xgboost.XGBRegressor().fit(X_train, y_train) if xgboost is not None else None', globals=globals()))

start = time.time()
GradientBoostingRegressor().fit(X_train, y_train)
elapsed_time = time.time() - start
print ("GradientBoostingRegressor_elapsed_time:{0} [sec]\n".format(elapsed_time))

print('------------------------------------------------------------------------------------------------------\n'
      '          7.6 stacking                                                                                \n'
      '------------------------------------------------------------------------------------------------------\n')
# Creating data set
X, y, coef = make_regression(random_state=12,
                       n_samples=100,
                       n_features=1,
                       n_informative=1,
                       noise=10.0,
                       bias=-0.0,
                       coef=True)

print('X = \n{0}\n'.format(X))
print()
print('y = \n{0}\n'.format(y))
print()
print('coef = \n{0}\n'.format(coef))
print()

plt.figure(figsize=(8, 4))
plt.title("Feature 1")
plt.plot(X, y, "bo")
plt.show()

# prepare data
X_train_valid, X_meta_valid, y_train_valid, y_meta_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.5, random_state=42)

# train base model
base_model_1 = LinearRegression()
base_model_2 = LGBMRegressor()
base_model_3 = KNeighborsRegressor()

base_model_1.fit(X_train, y_train)
base_model_2.fit(X_train, y_train)
base_model_3.fit(X_train, y_train)

# base predicts
base_pred_1 = base_model_1.predict(X_valid)
base_pred_2 = base_model_2.predict(X_valid)
base_pred_3 = base_model_3.predict(X_valid)

# test predicts for final result
valid_pred_1 = base_model_1.predict(X_meta_valid)
valid_pred_2 = base_model_2.predict(X_meta_valid)
valid_pred_3 = base_model_3.predict(X_meta_valid)

print ("mean squared error of model 1: {:.4f}".format(mean_squared_error(y_meta_valid, valid_pred_1)) )
print ("mean squared error of model 2: {:.4f}".format(mean_squared_error(y_meta_valid, valid_pred_2)) )
print ("mean squared error of model 3: {:.4f}".format(mean_squared_error(y_meta_valid, valid_pred_3)) )

# stack base predicts for training meta model
stacked_predictions = np.column_stack((base_pred_1, base_pred_2, base_pred_3))

# stack test predicts for final result
stacked_valid_predictions = np.column_stack((valid_pred_1, valid_pred_2, valid_pred_3))

# train meta model
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_valid)

# final result
meta_valid_pred = meta_model.predict(stacked_valid_predictions)
print ("mean squared error of meta model: {:.4f}\n".format(mean_squared_error(y_meta_valid, meta_valid_pred)))

