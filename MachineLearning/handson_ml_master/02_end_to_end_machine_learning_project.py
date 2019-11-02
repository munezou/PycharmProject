# To support both python 2 and python 3
#from __future__ import division, print_function, unicode_literals
'''
------------------------------------------------------------------------------------------------------------------------
Setup
------------------------------------------------------------------------------------------------------------------------
'''
# Common imports
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

print ('-------------------------------------------------------------------\n'
       '                           Setup                                   \n'
       '-------------------------------------------------------------------\n')

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
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

'''
------------------------------------------------------------------------------------------------------------------------
Get the data
------------------------------------------------------------------------------------------------------------------------
'''
print ('-------------------------------------------------------------------\n'
       '                         Get Data                                  \n'
       '-------------------------------------------------------------------\n')
import os
import tarfile
from six.moves import urllib

proxy_support = urllib.request.ProxyHandler({'https': 'http://proxy.kanto.sony.co.jp:10080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
pd.set_option('display.max_columns', None)
print('housing.head() = \n{0}'.format(housing.head()))
print()

housing.info()
print()

# Display ocean_proximity classification.
print('Display ocean_proximity classification = \n{0}'.format(housing["ocean_proximity"].value_counts()))
print()

print('housing.describe() = \n{0}'.format(housing.describe()))
print()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(15,10))
save_fig("attribute_histogram_plots")
plt.show()

# to make this notebook's output identical at every run
np.random.seed(42)

'''
------------------------------------------------------------------------------------------------------------------------
create test-set
------------------------------------------------------------------------------------------------------------------------
'''
print ('-------------------------------------------------------------------\n'
       '                     create test-set                               \n'
       '-------------------------------------------------------------------\n')
# to make this notebook's output identical at every run
np.random.seed(42)

import numpy as np

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    print('---< Sort data randomly. >---')
    shuffled_indices = np.random.permutation(len(data))
    print ('shuffled_indices = \n{0}'.format(shuffled_indices))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

print('---< Divide housing dataset to train_set and to test_set. >---')
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
print()

from zlib import crc32

'''
------------------------------------------------------------------------------------------------------------------------
zlib.crc32 (data [, value])
Calculate the CRC (Cyclic Redundancy Check) checksum of data. 
The result is an unsigned 32-bit integer. 
If value is given, it is used as the initial value for the checksum calculation. 
If not given, the default value of 1 is used. By giving value, you can calculate a checksum over the entire data that combines multiple inputs. 
This algorithm is not cryptographically strong and should not be used for authentication or digital signatures. Also, 
because it is designed as a checksum algorithm, it is not suitable for general-purpose hash algorithms.

Changed in version 3.0: Always returns an unsigned value. Use crc32 (data) & 0xffffffff to generate the same number across all versions and platforms of Python.
------------------------------------------------------------------------------------------------------------------------
'''

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

'''
------------------------------------------------------------------------------------------------------------------------
Divide to train_set and to test_set.
------------------------------------------------------------------------------------------------------------------------ 
'''
print ('-------------------------------------------------------------------\n'
       '               Divide to train_set and to test_set.                \n'
       '-------------------------------------------------------------------\n')
print('---< Divide By split_train_test_by_id. >---')
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

print('test_set.head() = \n{0}'.format(test_set.head()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Divide to train_set and to test_set.
------------------------------------------------------------------------------------------------------------------------ 
'''
print ('-------------------------------------------------------------------\n'
       '              Divide to train_set and to test_set.                 \n'
       '-------------------------------------------------------------------\n')
print('---< Divide By train_test_split. >---')
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print ('---< train_test_split of sklearn >---')
print('test_set.head() = \n{0}'.format(test_set.head()))
print()

housing["median_income"].hist()
plt.show()

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print('housing["income_cat"].value_counts() = \n{0}'.format(housing["income_cat"].value_counts()))
print()

housing["income_cat"].hist()
plt.show()
print()

print('---< splite by sklearn.model_selection.StratifiedShuffleSplit >---')
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# To compare housing["income_cat"] distribution and start_test_set["income_cat"] distribution.
# if housing["income_cat"] distribution is as same as start_test_set["income_cat"] distribution, to splite is  success.

test_set_byIncome = strat_test_set["income_cat"].value_counts() / len(strat_test_set)
housig_all_byIncome = housing["income_cat"].value_counts() / len(housing)

print('test_set_byIncome = \n{0}'.format(test_set_byIncome))
print()
print('housing_all_byIncome = \n{0}'.format(housig_all_byIncome))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Compare percentages by revenue category for the entire data set, test sets generated using stratified sampling, and test sets generated by random sampling.
------------------------------------------------------------------------------------------------------------------------
'''
print ('-------------------------------------------------------------------\n'
       ' Compare percentages by revenue category for the entire data set,  \n'
       ' test sets generated using stratified sampling,                    \n'
       ' and test sets generated by random sampling.                       \n'
       '-------------------------------------------------------------------\n')
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

# Output a calculation result.
print('comare_props = \n{0}'.format(compare_props))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Remove housing["income_cat"]
------------------------------------------------------------------------------------------------------------------------
'''
print("strat_train_set = \n{0}".format(strat_train_set.head()))
print()

# Here, the attribute of income_cat is removed and restored.
print('---< the attribute of income_cat is removed and restored. >---')
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

print('---< Remove incom_cat in strat_train_set. >---')
print("strat_train_set = \n{0}".format(strat_train_set.head()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Discover and visualize the data to gain insights
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------------\n'
       ' Discover and visualize the data to gain insights \n'
       '--------------------------------------------------\n')
#
housing = strat_train_set.copy()

# Saving figure bad_visualization_plot
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
plt.show()

# Change to express density shading.
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")
plt.show()

# California house prices
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
plt.show()
print()

# Add map
import matplotlib.image as mpimg
california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()
print()

'''
------------------------------------------------------------------------------------------------------------------------
Look for correlation.
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------------\n'
       '                Look for correlation.             \n'
       '--------------------------------------------------\n')
corr_matrix = housing.corr()

print('-----------------------------------------------------------------------------------\n'
      '                      correlation by median house price.                           \n'
      '-----------------------------------------------------------------------------------\n')

print('corr_matrix["median_house_value"].sort_values(ascending=False) = \n{0}'.format(corr_matrix["median_house_value"].sort_values(ascending=False)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Display the relationship of each item in a scatter diagram.
------------------------------------------------------------------------------------------------------------------------
'''
# Saving figure scatter_matrix_plot

# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
plt.show()
print()

# Create a distribution map of the median home price and median income with strong correlation.
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")
plt.show()
print()

'''
------------------------------------------------------------------------------------------------------------------------
Try a combination of attributes.
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------------\n'
       '         Try a combination of attributes.         \n'
       '--------------------------------------------------\n')
# Add combinations of attributes.
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
print('---< Try a combination of attributes. >---')
print('housing.head() = \n{0}'.format(housing.head()))
print()

print('---< retry to look for correlation. >---')
# retry to Look for correlation.
corr_matrix = housing.corr()
print('corr_matrix["median_house_value"].sort_values(ascending=False) = \n{0}'.format(corr_matrix["median_house_value"].sort_values(ascending=False)))
print()

#
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()
print()

print('housing.describe() = \n{0}'.format(housing.describe()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Prepare the data for Machine Learning algorithms
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------------\n'
       ' Prepare the data for Machine Learning algorithms \n'
       '--------------------------------------------------\n')

print('---< Prepare the data for Machine Learning algorithms >---')
# drop median_house_value. and set it to housing_label.
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

print('---< search null data. >---')
# search null in column.
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print('sample_incomplete_rows = \n{0}'.format(sample_incomplete_rows))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Clean data

The following three methods can be considered as a method of processing the missing value of total_bedrooms.
* Option 1: Remove the corresponding section.
* Option 2: Remove the entire attribute.
* Option 3: Set some value. (0, average, median, etc.)
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------\n'
       '                Clean data                  \n'
       '--------------------------------------------\n')
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print('sample_incomplete_rows = \n{0}'.format(sample_incomplete_rows))
print()

print ('---< clean data.(option 1) >---')
print('sample_incomplete_rows.dropna(subset=["total_bedrooms"])  = \n{0}'.format(sample_incomplete_rows.dropna(subset=["total_bedrooms"]) ))
print()
print ('---< clean data.(option 2) >---')
print('sample_incomplete_rows.drop("total_bedrooms", axis=1) = \n{0}'.format(sample_incomplete_rows.drop("total_bedrooms", axis=1) ))
print()
print ('---< clean data.(option 3) >---')
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
print('sample_incomplete_rows = \n{0}'.format(sample_incomplete_rows))
print()

'''
------------------------------------------------------------------------------------------------------------------------
scikit-learn has an Imputer class that can handle missing values well, so use it.
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------\n'
       ' how to use Imputer class in scikit-learn.  \n'
       '--------------------------------------------\n')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
print()

# Remove the text attribute because median can only be calculated on numerical attributes:
housing_num = housing.drop('ocean_proximity', axis=1)
print('husing_num.head() = \n{0}'.format(housing_num.head()))
print()

# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

print('imputer.statistics_ = \n{0}'.format(imputer.statistics_))
print()

# Check that this is the same as manually computing the median of each attribute:
print('housing_num.median().values = \n{0}'.format(housing_num.median().values))
print()

# Using a trained imputer, replace the missing value with the learned median and transform the training set.
X = imputer.transform(housing_num)

# Since X is a numpy array, convert it to pandas DataFRame.
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

print('housing_tr.loc[sample_incomplete_rows.index.values] = {0}'.format(housing_tr.loc[sample_incomplete_rows.index.values]))
print()

print('imputer.strategy = {0}'.format(imputer.strategy))
print()

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print('housing_tr.head() = \n{0}'.format(housing_tr.head()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Processing text / category attributes
Now let's preprocess the categorical input feature, ocean_proximity:
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------\n'
       '    Processing text / category attributes   \n'
       '--------------------------------------------\n')

housing_cat = housing[['ocean_proximity']]
print('housing_cat.head(10) = \n{0}'.format(housing_cat.head(10)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Use oneHotEncoder provided by scikit-learn to perform one-hot encoding.
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------\n'
      '                using sklearn.preprocessing.OrdinalEncoder                         \n'
      '-----------------------------------------------------------------------------------\n')

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print('type(housing_cat_encoded) = {0}'.format(type(housing_cat_encoded)))
print('housing_cat_encoded[:10] = \n{0}'.format(housing_cat_encoded[:10]))
print()

# output ordinary_encoding.categories_
print('ordinal_encoder.categories_ = \n{0}'.format(ordinal_encoder.categories_))
print()

print('-----------------------------------------------------------------------------------\n'
      '                using sklearn.preprocessing.OneHotEncoder                          \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print('type(housing_cat_1hot) = {0}'.format(type(housing_cat_1hot)))
print('housing_cat_1hot.toarray() = \n{0}'.format(housing_cat_1hot.toarray()))
print('cat_encoder.categories_ = \n{0}'.format(cat_encoder.categories_))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Let's create a custom transformer to add extra attributes:
------------------------------------------------------------------------------------------------------------------------
'''
# Display  columns of housing data-set.
print('-----------------------------------------------------------------------------------\n'
      '                     custom transform                                              \n'
      '-----------------------------------------------------------------------------------\n')
print('housing.columns = \n{0}'.format(housing.columns))
print()

from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

'''
notation)
 Alternatively, you can use Scikit-Learn's FunctionTransformer class that lets you easily create a transformer based on a transformation function (thanks to Hanmin Qin for suggesting this code). 
 Note that we need to set validate=False because the data contains non-float values (validate will default to False in Scikit-Learn 0.22).
'''
from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False, kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)

housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing.columns)+["rooms_per_household", "population_per_household"], index=housing.index)

print('housing_extra_attribs.head() = \n{0}'.format(housing_extra_attribs.head()))
print('housing_extra_attribs.head() = \n{0}'.format(housing_extra_attribs.head()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Pipeline:



StandardScaler:


------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------\n'
      '            How to use pipeline                                                    \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

print('housing_num_tr = \n{0}'.format(housing_num_tr))
print()

print('-----------------------------------------------------------------------------------\n'
      '              new model: housing_prepared                                          \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
print('housing_prepared = \n{0}'.format(housing_prepared))
print()
print('type(housing_prepared) = {0}'.format(type(housing_prepared)))
print('housing_prepared.shape = {0}'.format(housing_prepared.shape))
print()

print('-----------------------------------------------------------------------------------\n'
      '              old model: old_housing_prepared                                      \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.base import BaseEstimator, TransformerMixin
# Create a class to select numerical or categorical columns
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([('selector', OldDataFrameSelector(cat_attribs)), ('cat_encoder', OneHotEncoder(sparse=False)),])

from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", old_num_pipeline), ("cat_pipeline", old_cat_pipeline),])

old_housing_prepared = old_full_pipeline.fit_transform(housing)
print('old_housing_prepared = \n{0}'.format(old_housing_prepared))
print()
print('np.allclose(housing_prepared, old_housing_prepared) = {0}'.format(np.allclose(housing_prepared, old_housing_prepared)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
2.6 Select and train a model
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------\n'
      '                  Select and train a model (LinearRegression)                      \n'
      '-----------------------------------------------------------------------------------\n')
# LinearRegression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
some_lin_reg = lin_reg.predict(some_data_prepared)
list_some_labels = list(some_labels)

print('Predictions: \n{0}'.format(lin_reg.predict(some_data_prepared)))
print('     labels: \n{0}'.format(list(some_labels)))
print()
print("Labels:", list_some_labels)
print()

print('compare = \n{0}'.format(abs(some_lin_reg - list_some_labels)))
print()

print('some_data_prepared = \n{0}'.format(some_data_prepared))
print()

print('-----------------------------------------------------------------------------------\n'
      '                     Evaluate the training set.                                    \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('lin_rmse = {0}'.format(lin_rmse))
print()

from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print('lin_mae = {0}'.format(lin_mae))
print()

print('-----------------------------------------------------------------------------------\n'
      '               Select and train a model (DecisionTreeRegressor)                    \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

some_data_prepared = full_pipeline.transform(some_data)

print('Predictions: \n{0}'.format(tree_reg.predict(some_data_prepared)))
print('     labels: \n{0}'.format(list(some_labels)))
print()
print('some_data_prepared = \n{0}'.format(some_data_prepared))
print()

# Model validation of DecisionTreeRegressor
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print('tree_rmse = {0}'.format(tree_rmse))
print()

'''
------------------------------------------------------------------------------------------------------------------------
            2.6.2 Better evaluation using cross-validation
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------------\n'
       '  2.6.2 Better evaluation using cross-validation  \n'
       '--------------------------------------------------\n')
from sklearn.model_selection import cross_val_score

print('---< in case of DecisionTreeRegressor model >---')
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
print()

print('-----------------------------------------------------------------------------------\n'
      '            Evaluation using cross-validation(LinearRegression)                    \n'
      '-----------------------------------------------------------------------------------\n')
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
print()

print('-----------------------------------------------------------------------------------\n'
      '           Evaluation using cross-validation(RandomForestRegressor)                \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print('forest_rmse = {0}'.format(forest_rmse))
print()

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
print()

print('-----------------------------------------------------------------------------------\n'
      '            Evaluation using cross-validation(LinearRegression)                    \n'
      '-----------------------------------------------------------------------------------\n')
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_reg_describe = pd.Series(np.sqrt(-scores)).describe()
print('lin_reg_describe = \n{0}'.format(lin_reg_describe))
print()

print('-----------------------------------------------------------------------------------\n'
      '                 Evaluation using cross-validation(SVR)                            \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print('svm_rmse = {0}'.format(svm_rmse))
print()

'''
------------------------------------------------------------------------------------------------------------------------
2.7 Fine tune the model.
------------------------------------------------------------------------------------------------------------------------
'''
print ('--------------------------------------------------\n'
       '                 2.7.1 grid search                \n'
       '--------------------------------------------------\n')
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print('grid_search.best_params_ = {0}'.format(grid_search.best_params_))
print()

print('grid_search.best_estimator_ = \n{0}'.format(grid_search.best_estimator_))
print()

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print('pd.DataFrame(grid_search.cv_results_) = \n{0}')
print()

print('---< best parameter and best score >---')
print('best_parameter = {0}'.format(grid_search.best_params_))
print('best_score = {0}'.format(grid_search.best_score_))
print()
print('---< Compare housing_prepared and housing_labels. >--- ')
predictedValue = pd.DataFrame(grid_search.predict(housing_prepared))
housing_labels_pd = pd.DataFrame(housing_labels)
print('predictedValue = \n{0}'.format(predictedValue.head()))
print()
print('housing_labels_pd = \n{0}'.format(housing_labels_pd.head()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
2.7.2 Randam Search
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------\n'
      '                             2.7.2 Randam Search                                   \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print()
cvres_best_parameter = rnd_search.best_params_
print('cvres_best_parameter = {0}'.format(cvres_best_parameter))
print()
cvres_best_score = rnd_search.best_score_
print('cvres_best_score = {0}'.format(cvres_best_score))
print()
print('---< Compare housing_prepared and housing_labels. >--- ')
predictedValue = pd.DataFrame(rnd_search.predict(housing_prepared))
housing_labels_pd = pd.DataFrame(housing_labels)
print('predictedValue = \n{0}'.format(predictedValue.head()))
print()
print('housing_labels_pd = \n{0}'.format(housing_labels_pd.head()))
print()

print('---< feature importances >---')
feature_importances = grid_search.best_estimator_.feature_importances_
print('feature_importances = \n{0}'.format(feature_importances))
print()

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print('sorted = \n{0}'.format(sorted(zip(feature_importances, attributes), reverse=True)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
2.7.5 Evaluate the system with a test set.
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------\n'
      '                  2.7.5 Evaluate the system with a test set.                       \n'
      '-----------------------------------------------------------------------------------\n')
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('final_rmse = {0}'.format(final_rmse))
print()

print('---< We can compute a 95% confidence interval for the test RMSE: >---')
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

confidence_interval = np.sqrt(stats.t.interval(confidence, m - 1, loc=np.mean(squared_errors), scale=stats.sem(squared_errors)))
print('95% confidence interval = {0}'.format(confidence_interval))
print()

print('---< We could compute the 95% confidence interval manually like this: >---')
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
print('95% confidence interval = ({0}, {1})'.format(np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)))
print()
print('---< Alternatively, we could use a z-scores rather than t-scores: >---')
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
print('95% confidence interval = ({0}, {1})'.format(np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
2.7.6 Full system operation, monitoring and maintenance.
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------\n'
      '            2.7.6 Full system operation, monitoring and maintenance.               \n'
      '-----------------------------------------------------------------------------------\n')
print('---< A full pipeline with both preparation and prediction >---')
full_pipeline_with_predictor = Pipeline([("preparation", full_pipeline), ("linear", LinearRegression())])

full_pipeline_with_predictor.fit(housing, housing_labels)
result_predict = full_pipeline_with_predictor.predict(some_data)
print('result_predict = \n{0}'.format(result_predict))
print()

print('---< Model persistence using joblib >---')
my_model = full_pipeline_with_predictor

from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF
print()

print('---< Example SciPy distributions for RandomizedSearchCV >---')
from scipy.stats import geom, expon
geom_distrib=geom(0.5).rvs(10000, random_state=42)
expon_distrib=expon(scale=1).rvs(10000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()

plt.hist(expon_distrib, bins=50)
plt.show()

print()

'''
------------------------------------------------------------------------------------------------------------------------
2.10 Exercise solutions
------------------------------------------------------------------------------------------------------------------------
'''
'''
Question 1: 
Try a Support Vector Machine regressor (sklearn.svm.SVR), 
with various hyperparameters such as kernel="linear" (with various values for the C hyperparameter) 
or kernel="rbf" (with various values for the C and gamma hyperparameters). 
Don't worry about what these hyperparameters mean for now. How does the best SVR predictor perform?
'''
print('-----------------------------------------------------------------------------------\n'
      '   2.10 Question 1: Try a Support Vector Machine regressor (sklearn.svm.SVR)       \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(housing_prepared, housing_labels)
print()
print('---< The best model achieves the following score (evaluated using 5-fold cross validation): >---')
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
print('rmse = {0}'.format(rmse))
print()
# That's much worse than the RandomForestRegressor.
print('---< Let us check the best hyperparameters found: >---')
print('grid_search.best_params_ = {0}'.format(grid_search.best_params_))
print()
print('-----------------------------------------------------------------------------------\n'
      '     2.10 Question 2: Try replacing GridSearchCV with RandomizedSearchCV.          \n'
      '-----------------------------------------------------------------------------------\n')
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs, n_iter=50, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

print('---< The best model achieves the following score (evaluated using 5-fold cross validation): >---')
negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
print('rmse = {0}'.format(rmse))
print()
# Now this is much closer to the performance of the RandomForestRegressor (but not quite there yet).
print('---< Let us check the best hyperparameters found: >---')
print('the best hyperparameters = {0}'.format(rnd_search.best_params_))
print()

'''
This time the search found a good set of hyperparameters for the RBF kernel.
Randomized search tends to find better hyperparameters than grid search in the same amount of time.
'''

'''
------------------------------------------------------------------------------------------------------------------------
Let's look at the exponential distribution we used, with scale=1.0. 
Note that some samples are much larger or smaller than 1.0, but when you look at the log of the distribution, 
you can see that most values are actually concentrated roughly in the range of exp(-2) to exp(+2), which is about 0.1 to 7.4.
------------------------------------------------------------------------------------------------------------------------
'''
print('---< feature of expon (scale = 1.0) used for gamma >---')
expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()

print()

'''
------------------------------------------------------------------------------------------------------------------------
The distribution we used for C looks quite different: the scale of the samples is picked from a uniform distribution within a given range,
which is why the right graph, which represents the log of the samples, looks roughly constant. 
This distribution is useful when you don't have a clue of what the target scale is:
------------------------------------------------------------------------------------------------------------------------
'''
print('---< feature of reciprocal(20, 200000) used for C >---')
reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()

print()

print('-----------------------------------------------------------------------------------------------------------------------\n'
      '  2.10 Question 3: Try adding a transformer in the preparation pipeline to select only the most important attributes.  \n'
      '-----------------------------------------------------------------------------------------------------------------------\n')
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

'''
Note: 
this feature selector assumes that you have already computed the feature importances somehow (for example using a RandomForestRegressor). 
You may be tempted to compute them directly in the TopFeatureSelector's fit() method, 
however this would likely slow down grid/randomized search since the feature importances would have to be computed for every hyperparameter combination (unless you implement some sort of cache).
'''
print('---< Let us define the number of top features we want to keep: >---')
k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
print('top_k_feature_indices = \n{0}'.format(top_k_feature_indices))
print()
print('np.asarray(attributes)[top_k_feature_indices] = \n{0}'.format(np.asarray(attributes)[top_k_feature_indices]))
print()

print('---< Let us double check that these are indeed the top k features: >---')
sorted_result = sorted(zip(feature_importances, attributes), reverse=True)[:k]
print('sorted_result = \n{0}'.format(sorted_result))
print()

print('---< let us create a new pipeline that runs the previously defined preparation pipeline, and adds top k feature selection: >---')
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)

print('---< Let us look at the features of the first 3 instances: >---')
print('housing_prepared_top_k_features[0:3] = \n{0}'.format(housing_prepared_top_k_features[0:3]))
print()
print('housing_prepared[0:3, top_k_feature_indices] = \n{0}'.format(housing_prepared[0:3, top_k_feature_indices]))
print()

print('-----------------------------------------------------------------------------------------------------------------------\n'
      '    2.10 Question 4: Try creating a single pipeline that does the full data preparation plus the final prediction.     \n'
      '-----------------------------------------------------------------------------------------------------------------------\n')
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])

prepare_select_and_predict_pipeline.fit(housing, housing_labels)

print('---< Let us try the full pipeline on a few instances: >---')
some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))
print()

print('--------------------------------------------------------------------------------------------------------------\n'
      '               2.10 Question 5: Automatically explore some preparation options using GridSearchCV.            \n'
      '--------------------------------------------------------------------------------------------------------------\n')
param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search_prep.fit(housing, housing_labels)

print('grid_search_prep.best_params_ = \n{0}'.format(grid_search_prep.best_params_))
print()

'''
The best imputer strategy is most_frequent and apparently almost all features are useful (15 out of 16). 
The last one (ISLAND) seems to just add some noise.
'''
