###On hands on machine learning with tensorflow and scikit learning.
##Exercises chapter 2:
#Data cleaning steps from chapter 2 look at chapter to see full 
#anotations.
import pandas as pd
import matplotlib.pyplot as plt 
housing = pd.read_csv("/Users/masonkarsevar/housing.csv")

import numpy as np 
import hashlib 

def test_set_check(identifier, test_ratio, hash):
	return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash= hashlib.md5):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
	return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
print(housing["income_cat"].describe())
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis = 1, inplace = True)

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.preprocessing import Imputer

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room = True): # No *args or **kargs
		self.add_bedrooms_per_room = add_bedrooms_per_room 
	def fit(self, X, y = None):
		return self #Nothing else to do. most likely because of the BaseEstimator
		#parent class. This is where the kargs and args will be defined.
	def transform(self, X, y = None):
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, 
						bedrooms_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin 
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names 
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values

housing_num = housing.drop("ocean_proximity", axis = 1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
	("Selector", DataFrameSelector(num_attribs)),
	("imputer", Imputer(strategy = "median")),
	("attribs_adder", CombinedAttributesAdder()),
	("std_scaler", StandardScaler()),
])

import python_ML_cat_encoder as en# the following module was pulled from 
#ageron's github. Don't use the onehotencoder for this problem.

cat_pipeline = Pipeline([
	("selector", DataFrameSelector(cat_attribs)),
	("cat_encoder", en.CategoricalEncoder(encoding = "onehot-dense")),
])

from sklearn.pipeline import FeatureUnion 

full_pipeline = FeatureUnion(transformer_list = [
	("num_pipeline", num_pipeline),
	("cat_pipeline", cat_pipeline),
	])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

##Problem 1:
from sklearn import svm
from sklearn.model_selection import GridSearchCV
param_grid = [
    {"kernel": ["linear"],"C": [40, 50, 60, 70]},
]
svr = svm.SVR()
grid_search = GridSearchCV(svr, param_grid, cv = 5,
            scoring = "neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)
#To Save computation time, for this problem I used the linear
#kernel in place of the "rbf" kernel. 
print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
#Interestingly it seems that the mean squared error rate decreases with 
#increased hyperparameter C for Support vector machines. Will need 
#to experiment with higher C values 
#The best MSE value for model SVR(kernel = "linear", C = 40) is 
#74097 this is still more than the random forest method.  

#through increasing the gridsearchCV param_grid method to 70 there
#the mse value descreased further to 72325. I think this is the best
#the linear SVM.svr can do for this dataset.

##problem 2: (Using RandomizedSearchCV in place of GridSearchCV)
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm

svr = svm.SVR()
fit_params = [
    {"kernel":["linear","rbrf"],
    "gamma": [0.1,0.6], "C": [1, 20]}
    ]
rand_grid = RandomizedSearchCV(cv=5, error_score = "raise", 
    estimator=svm.SVR(),
    random_state=42, refit=True, scoring="neg_mean_squared_error")
rand_grid.fit(housing_prepared, housing_labels, fit_params)
cvres = rand_grid.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

##problem 4:
#author's solution: (with some modifications).
from sklearn import svm
from sklearn.metrics import mean_squared_error 

prepare_select_and_predict_pipeline = Pipeline([
	("preparation", full_pipeline),
	("svr_model", svm.SVR(kernel="linear", C=70)), 
])
housing_svr = prepare_select_and_predict_pipeline.fit(housing, housing_labels)

housing_test_x = strat_test_set.drop("median_house_value", axis=1)
housing_test_y = strat_test_set["median_house_value"].copy()

housing_test_pred = housing_svr.predict(housing_test_x)
print(np.sqrt(mean_squared_error(housing_test_y, housing_test_pred)))









