# common library
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt



print('--------------------------------------------------------------------------------------------------------------\n'
      ' 7.                                                                                                           \n'
      '  Exercise: train and fine-tune a Decision Tree for the moons dataset.                                        \n'
      '--------------------------------------------------------------------------------------------------------------\n')
print('---< a. Generate a moons dataset using make_moons(n_samples=10000, noise=0.4). >---')

# create make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# plot raw data
plt.figure(figsize=(8, 7))
plt.title("moons(noise=0.4, randam_state=42)")
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c='red', label="X0")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c='blue', label="X1")
plt.grid(True)
plt.legend()
plt.show()

print('---< b. Split it into a training set and a test set using train_test_split(). >---')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('--------------------------------------------------------------------------------------------------------------\n'
      ' c. Use grid search with cross-validation (with the help of the GridSearchCV class)                           \n'
      ' to find good hyperparameter values for a DecisionTreeClassifier.                                             \n'
      ' Hint: try various values for max_leaf_nodes.                                                                 \n'
      '--------------------------------------------------------------------------------------------------------------\n')
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3)

grid_search_cv_fit = grid_search_cv.fit(X_train, y_train)

grid_search_cv_best_estimator = grid_search_cv.best_estimator_
print('grid_search_cv_best_estimator = \n{0}\n'.format(grid_search_cv_best_estimator))

print('--------------------------------------------------------------------------------------------------------------\n'
      ' d. Train it on the full training set using these hyperparameters,                                            \n'
      ' and measure your model of performance on the test set. You should get roughly 85% to 87% accuracy.           \n'
      ' By default, GridSearchCV trains the best model found on the whole training set                               \n'
      '(you can change this by setting refit=False), so we do not need to do it again.                               \n'
      'We can simply evaluate the model of accuracy:                                                                 \n'
      '--------------------------------------------------------------------------------------------------------------\n')
y_pred = grid_search_cv.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('acc = {0}\n'.format(acc))

print('--------------------------------------------------------------------------------------------------------------\n'
      ' 8.                                                                                                           \n'
      '  Exercise: Grow a forest.                                                                                    \n'
      '--------------------------------------------------------------------------------------------------------------\n')
print()

print('--------------------------------------------------------------------------------------------------------------\n'
      ' a. Continuing the previous exercise, generate 1,000 subsets of the training set,                             \n'
      '    each containing 100 instances selected randomly.                                                          \n'
      '  Hint: you can use Scikit-Learn of ShuffleSplit class for this.                                              \n'
      '--------------------------------------------------------------------------------------------------------------\n')
n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

print('--------------------------------------------------------------------------------------------------------------\n'
      ' b. Train one Decision Tree on each subset, using the best hyperparameter values found above.                 \n'
      '    Evaluate these 1,000 Decision Trees on the test set.                                                      \n'
      ' Since they were trained on smaller sets,                                                                     \n'
      ' these Decision Trees will likely perform worse than the first Decision Tree,                                 \n'
      ' achieving only about 80% accuracy.                                                                           \n'
      '---------------------------------------------------------------------------------------------------------------\n')
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)

    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print('np.mean(accuracy_scores) = {0}\n'.format(np.mean(accuracy_scores)))

print('--------------------------------------------------------------------------------------------------------------\n'
      ' c. Now comes the magic. For each test set instance, generate the predictions of the 1,000 Decision Trees,    \n'
      '    and keep only the most frequent prediction (you can use SciPy of mode() function for this).               \n'
      '    This gives you majority-vote predictions over the test set.                                               \n'
      '---------------------------------------------------------------------------------------------------------------\n')
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

print('y_pred_majority_votes = {0}\n, n_votes = {1}\n'.format(y_pred_majority_votes, n_votes))
print()

print('--------------------------------------------------------------------------------------------------------------\n'
      ' d. Evaluate these predictions on the test set:                                                               \n'
      '    you should obtain a slightly higher accuracy than your first model (about 0.5 to 1.5% higher).            \n'
      '    Congratulations, you have trained a Random Forest classifier!                                             \n'
      '---------------------------------------------------------------------------------------------------------------\n')
acc = accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
print('acc = {0}\n'.format(acc))
