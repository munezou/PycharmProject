# common library
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform


print('---------------------------------------------------------------------------------------------------------------\n'
      ' 10.                                                                                                           \n'
      '  train an SVM regressor on the California housing dataset.                                                    \n'
      '  Let us load the dataset using fetch_california_housing() function of Scikit-Learn:                           \n'
      '---------------------------------------------------------------------------------------------------------------\n')
# get fetch_california_housing()
housing = fetch_california_housing()

# housing information
print('housing information = \n{0}'.format(housing.DESCR))
print()

X = housing["data"]
y = housing["target"]

# Split it into a training set and a test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Don't forget to scale the data:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_svr = LinearSVR(random_state=42)
lin_svr_fit = lin_svr.fit(X_train_scaled, y_train)
print('lin_svr_fit = \n{0}'.format(lin_svr_fit))
print()

# Let's see how it performs on the training set:
y_pred = lin_svr.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
print('mse = {0}'.format(mse))

# Let's look at the RMSE:
np_sqrt = np.sqrt(mse)
print('np_sqrt = {0}'.format(np_sqrt))
print()

'''
------------------------------------------------------------------------------------------------------------------------
In this training set, the targets are tens of thousands of dollars. 
The RMSE gives a rough idea of the kind of error you should expect (with a higher weight for large errors): 
so with this model we can expect errors somewhere around $10,000. Not great. 
Let's see if we can do better with an RBF Kernel. 
We will use randomized search with cross validation to find the appropriate hyperparameter values for C and gamma:
------------------------------------------------------------------------------------------------------------------------
'''
param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, cv=3, random_state=42)
rnd_search_cv_fit = rnd_search_cv.fit(X_train_scaled, y_train)
print('rnd_search_cv_fit = \n{0}'.format(rnd_search_cv_fit))
print()

print('rnd_search_cv.best_estimator_ = {0}'.format(rnd_search_cv.best_estimator_))
print()

# Now let's measure the RMSE on the training set:
y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
np_sqrt = np.sqrt(mse)
print('np_sqrt = {0}'.format(np_sqrt))
print()

# Looks much better than the linear model.
# Let's select this model and evaluate it on the test set:
y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print('np_sqrt(test) = {0}'.format(np_sqrt))
print()