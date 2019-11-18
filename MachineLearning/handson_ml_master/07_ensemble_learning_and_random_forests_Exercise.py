# Common imports
import numpy as np

from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score


print('--------------------------------------------------------------------------------------------------------------\n'
      ' 8. Voting Classifier                                                                                         \n'
      '  Exercise: Load the MNIST data and split it into a training set,                                             \n'
      '            a validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for validation, \n'
      '            and 10,000 for testing).                                                                          \n'
      '--------------------------------------------------------------------------------------------------------------\n')
# read minist dataset
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)

# Secure 10,000 test data.
X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)

# Secure 10,000 Validation data and 50,000 train data.
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

# Exercise: Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM.
random_forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
svm_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

# fitting
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print('Training the {0}\n'.format(estimator))
    estimator.fit(X_train, y_train)

'''
The linear SVM is far outperformed by the other classifiers. 
However, let's keep it for now since it may improve the voting classifier's performance.
'''

'''
Exercise: Next, 
 try to combine them into an ensemble that outperforms them all on the validation set, 
 using a soft or hard voting classifier.
'''

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

# ensemble classifier is VotingClassifier.
voting_clf = VotingClassifier(named_estimators)

# fitting by train data(50,000)
voting_clf_fit = voting_clf.fit(X_train, y_train)
print('voting_clf_fit = \n{0}\n'.format(voting_clf_fit))

# Calculate score of voting classifier by Validation data(10,000)
voting_clf_score = voting_clf.score(X_val, y_val)
print('voting_clf_score = {0}\n'.format(voting_clf_score))

# Calculate the score value of each classifier used to voting classfiers.
estimators_score = [estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
print('estimators_score = \n{0}\n'.format(estimators_score))

'''
Let's remove the SVM to see if performance improves. 
It is possible to remove an estimator by setting it to None using set_params() like this:
'''

# remove svm_clf
voting_clf_modify = voting_clf.set_params(svm_clf=None)
print('voting_clf_modify = \n{0}\n'.format(voting_clf_modify))

# This updated the list of estimators:
print('voting_clf.estimators = \n{0}\n'.format(voting_clf.estimators))

# However, it did not update the list of trained estimators:
print('voting_clf.estimators_ = \n{0}\n'.format(voting_clf.estimators_))

# So we can either fit the VotingClassifier again, or just remove the SVM from the list of trained estimators:
del voting_clf.estimators_[2]

# confirm a condition after remved SVM.
print('voting_clf.estimators_ = \n{0}\n'.format(voting_clf.estimators_))

# Now let's evaluate the VotingClassifier again:
voting_clf_modify_score = voting_clf.score(X_val, y_val)
print('voting_clf_modify_score = {0}\n'.format(voting_clf_modify_score))

'''
A bit better! 
The SVM was hurting performance. 
Now let's try using a soft voting classifier. 
We do not actually need to retrain the classifier, we can just set voting to "soft":
'''

voting_clf.voting = "soft"

voting_clf_soft_score = voting_clf.score(X_val, y_val)
print('voting_clf_soft_score = {0}\n'.format(voting_clf_soft_score))

'''
That's a significant improvement, and it's much better than each of the individual classifiers.

Once you have found one, try it on the test set. How much better does it perform compared to the individual classifiers?
'''
voting_clf_soft_score_test = voting_clf.score(X_test, y_test)
print('voting_clf_soft_score = {0}\n'.format(voting_clf_soft_score_test))

voting_clf_modify_estimator_score = [estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]
print('voting_clf_modify_estimator_score = \n{0}\n'.format(voting_clf_modify_estimator_score))

'''
The voting classifier reduced the error rate from about 4.0% for our best model (the MLPClassifier) to just 3.1%. 
That's about 22.5% less errors, not bad!
'''

print('--------------------------------------------------------------------------------------------------------------\n'
      ' 9. Stacking Ensemble                                                                                         \n'
      '  Exercise:                                                                                                   \n'
      '  Run the individual classifiers from the previous exercise to make predictions on the validation set,        \n'
      '  and create a new training set with the resulting predictions:                                               \n'
      '  each training instance is a vector containing the set of predictions from all your classifiers for an image,\n'
      '  and the target is the image of class. Train a classifier on this new training set.                          \n'
      '--------------------------------------------------------------------------------------------------------------\n')
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

print('X_val_predictions = \n{0}\n'.format(X_val_predictions))

rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender_fit = rnd_forest_blender.fit(X_val_predictions, y_val)
print('rnd_forest_blender_fit = \n{0}\n'.format(rnd_forest_blender_fit))

print('rnd_forest_blender.oob_score_ = {0}\n'.format(rnd_forest_blender.oob_score_))

'''
------------------------------------------------------------------------------------------------------------------
You could fine-tune this blender or try other types of blenders (e.g., an MLPClassifier),
then select the best one using cross-validation, as always.

Exercise: Congratulations, you have just trained a blender,
 and together with the classifiers they form a stacking ensemble! Now let's evaluate the ensemble on the test set. 
 For each image in the test set, make predictions with all your classifiers, 
 then feed the predictions to the blender to get the ensemble's predictions. 
 How does it compare to the voting classifier you trained earlier?
 -----------------------------------------------------------------------------------------------------------------
'''
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

y_pred = rnd_forest_blender.predict(X_test_predictions)
print('accuracy_score(y_test, y_pred) = {0}\n'.format(accuracy_score(y_test, y_pred)))

'''
--------------------------------------------------------------------------------------------------------------------
This stacking ensemble does not perform as well as the soft voting classifier we trained earlier, 
it's just as good as the best individual classifier.
--------------------------------------------------------------------------------------------------------------------
'''