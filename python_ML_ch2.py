###Honds-on machine learning with sciki learning and tensorflow:
##Chapter 2 End to end machine learning project:
#Repositories for up and coming data scientists:
#http://archive.ics.uci.edu/ml/
#https://kaggle.com/datasets/
#https://aws.amazon.com/fr/datasets/

##Meta portals 
#https://dataportals.org/
#http://opendatamonitor.eu/
#http://quandl.com/

##other pages listing many popular open data repositories:
#wikipedia's list of machine learning datasets https://goo.gl/SJHN2K
#http://goo.gl/zDR78y
#https://www.reddit.com/r/datasets

#for this chapter the author uses a california housing data set from the bay area 
#based in the 1990s from the us census bureau.

##Looking at the big picture:
##Frame the Problem:
#(create a learning algorithm that predicts the district median housing price).

##Find the appropriate learning algorithm for the question at hand:
#Since the data set is real estate price evaluation and the data set used 
#to train the learning algorithm will be relatively small. Multivariate 
#regression will be more than adequate combined with batch cross validation.

##Select a Performance measure:
#the typical performance measure is Root mean square error: look at page 
#37 to see more details.
#But for datasets that are riddled with outliers professionals use the 
#mean absolute error (look at page 39 for the equation).

#It seems that I couldn't get the following command:
import os 
import tarfile 
from six.moves import urllib 

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/blob/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	if not os.path.isdir(housing_path):
		os.makedirs(housing_path)
	tgz_path = os.path.join(housing_path, "housing.tgz")
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = targile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()

import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)

#to work hence I used the panda command read.csv("housing.csv") to read 
#in the csv file into juypter. Honestly juypter is almost like R in many ways.

##plotting the data as a histogram:
#This is like the pairs() function in R:
import pandas as pd 
import matplotlib.pyplot as plt
housing = pd.read_csv("/Users/masonkarsevar/housing.csv")
housing.hist(bins = 50, figsize = (20, 15))
#plt.show()#It seems like matplotlib.pyplot only works with sublime text 
#on my system. this is another problem that I will have to fix on my system.

#I take that back this command works just fine on juypter.

##Creating a Test set:
#Creating a test set is theoretically quite simple: just pick some instances 
#randomly, typically 20 percent of the dataset and set them aside.
import numpy as np 

def split_train_test(data, test_ratio):
	shuffle_indices = np.random.permutation(len(data))
	test_set_size = int(len(data)* test_ratio)
	test_indices = shuffle_indices[:test_set_size]
	train_indices = shuffle_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]
#the R alternative is use sample(1:nrow(housing), 
#as.integer(nrow(housing)*test_ratio), replacement = FALSE)

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +" , len(test_set), "test")

#Making sure that the learning algorithm does not see the entirety of your 
#test set.
#you can compute a hash of each instance's identifier, keep only the last 
#byte of the hash, and put the instance in the test set if this value is lower
#or equal to 51(~20 percent of 256)

import hashlib 

def test_set_check(identifier, test_ratio, hash):
	return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash= hashlib.md5):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
	return data.loc[~in_test_set], data.loc[in_test_set]

#Using the row index as the id:
housing_with_id = housing.reset_index() #adds index column 
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#for a more stable test and training set identifier use the latitude and 
#longitude:
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
#the problem with this method is that it introduces sampling bias since the 
#many districts will have the same id.

##Scikit learn has the function train_test_split that has the same functionality
#as the function that the author wrote except that there is an additional 
#argument that allows you to set the random seed and a join argument if you
#want to sample two different datasets from the same data base structure.

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

##Transfroming the income variable into five different discrete variable 
#categories:
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
print(housing["income_cat"].describe())#It seems that before the succeeding 
#command there are 11 different cateogories. Interesting method but this does
#go against the author of "Statistics Done Wrong".
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

#Stratified sampling based on the income category:
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

#The data sets generated from stratified sampling:
print(housing["income_cat"].value_counts() / len(housing))#This tells us the 
#percent distribution of the income data within the housing dataset. 
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))
#No way the author was right the distributions for the training and testing
#sets are all identical. I said never thought of this in my R code before.
#This is genius.

#The data sets generated from random sampling:
train_set_ran, test_set_ran = train_test_split(housing, test_size = 0.2, random_state = 42)
print(train_set_ran["income_cat"].value_counts() / len(train_set_ran))
print(test_set_ran["income_cat"].value_counts() / len(test_set_ran))
#Interesting the random training and testing sets are indeed skewed but not 
#by much. Will need to look into how this can mess with a learning algorithm.

##Converting the income_cat category to its original state:
for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis = 1, inplace = True)

##Discover and visualize the data to gain insights:
housing = strat_train_set.copy()

##Visualizing Geographical data:
housing.plot(kind = "scatter", x = "longitude", y = "latitude")
#plt.show() #Really cool this is just like my quake R dataset. Will need to see if
#I can do some three dimensional data visualizations to this dataset. 

#Changing the alpha value to visualize places that have higher population 
#density:
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)
#plt.show()#much better!!!

#cool the author is using the color and and point type arguments to illustrate 
#price and population differences. this is just like R for data science and 
#tidyverse.
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4, 
	s = housing["population"]/100, label = "population", figsize = (10, 7),
	c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True,
)
plt.legend()
#plt.show()#This module is very efficient. This visualization looks very good!!

##Looking for Correlations:
#computing the standard correlation coefficient (Pearson's r) between every pair of
#attributes using the corr() method.
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending = False))

#Using the scatter plot matrix to find correlatin between variables:
from pandas.plotting import scatter_matrix 

attributes = ["median_house_value", "median_income", "total_rooms",
			"housing_median_age"]
scatter_matrix(housing[attributes], figsize = (12, 8))
#plt.show()

#Exploring the correlation between median_income and median_house_value.
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value",
	alpha = 0.1)
#plt.show()

##Experimenting with Attribute Combinations:
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
#will need to remember this technique with my baseball data.

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

##Prepare the data for machine learning algorithms:
housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

##Data cleaning:
#Dealing with na values in python:
housing.dropna(subset = ["total_bedrooms"])#dropping the na values from the 
#total_bedrooms variable. 
housing.drop("total_bedrooms", axis = 1)#dropping the entire variable from 
#the data set.
median = housing["total_bedrooms"].median()#imputation using the median of 
#the variable.
housing["total_bedrooms"].fillna(median, inplace = True)

#You can also impute the total bedrooms column through the imputer method 
#in the scikit learning package.
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy = "median")
housing_num = housing.drop("ocean_proximity", axis = 1)#removing the categorical
#variable ocean proximity to fit the imputed median value to the data set.
imputer.fit(housing_num)#this imputes all of the numeric values in the dataset 
#if an na value exists. This is extremely useful.
print(imputer.statistics_)
print(housing_num.median().values)

x = imputer.transform(housing_num)

#putting the features in housing_num back into housing.
housing_tr = pd.DataFrame(x, columns = housing_num.columns)

##Handling Text and categorical attributes:
#Converting discrete strings into numerical variables.
from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)

#I believe that the author wants to change this into a binary categorical
#variable hence to carry out this task he will use the OneHotEncoder module.
from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
#print(housing_cat_1hot)#Now I understand the onehotencoder assigns values to 
#the categoritical variables ocean proximity. Will need to look for into this.

#to see the onehotencoder sparse matrix as a numpy array.
#print(housing_cat_1hot.toarray())

#these steps can be condensed into one through the sklearn labelBinarizer module.
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer() 
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
#The numpy array was created automatically.
#Sparse matrixes can be created through setting the command:
sparse_encoder = LabelBinarizer(sparse_output = True)
print(sparse_encoder.fit_transform(housing_cat))

##Custom Transformers:
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

##Feature Scaling:
#the author talks about standardization scaling (namely z-scores) and normalization
#scaling (like the method used by introduction to statistical learning).
#standardization scaler function standardscaler() and normalization scaler
#MinMaxScaler().

##Transformation Pipelines:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
	("imputer", Imputer(strategy= "median")),
	("attribs_adder", CombinedAttributesAdder()),
	("std_scaler", StandardScaler()),
])

#housing_num_tr = num_pipeline.fit_transform(housing_num)
#print(housing_num_tr)
#Will need to go back to this page. this will be very useful for future projects
#page 67.

#This transforms the dataframe into a numpy array for the pipeline methods 
#defined above:
from sklearn.base import BaseEstimator, TransformerMixin 
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names 
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values 

##Creating pipeline data processing steps for categorical and numeric variables
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

#Combining the results from the two pipelines:
from sklearn.pipeline import FeatureUnion 

full_pipeline = FeatureUnion(transformer_list = [
		("num_pipeline", num_pipeline),
		("cat_pipeline", cat_pipeline),
		])

#running the full pipeline coded above:
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

##Training and Evaluating on the Training set:
#Linear regression model:
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

#RMSE estimation for the training set:
from sklearn.metrics import mean_squared_error 

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)#The mean squared error rate was estimated at 
#68628.

#Decision tree model:
from sklearn.tree import DecisionTreeRegressor 

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

##Better Evaluation Using Cross Validation: This examples uses 
#k fold cross validation in place of LOOCV.
from sklearn.model_selection import cross_val_score 

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
						scoring="neg_mean_squared_error", cv = 10)
tree_rmse_scores = np.sqrt(-scores)

#Cross validation results of the decision tree method:
def display_scores(scores):
	print("Scores:", scores)
	print("mean:", scores.mean())
	print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
#The decision tree method obtained a rmse error rate of 
#71177 (worse than the linear regression method).

#Cross validation with linear regression:
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
							scoring = "neg_mean_squared_error", cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(display_scores(lin_rmse_scores))
#the Linear regression method performs a little better.

#Random Forest model experiment:
from sklearn.ensemble import RandomForestRegressor 

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
print(forest_mse)

#cross validation step:
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
							scoring = "neg_mean_squared_error", cv = 10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(display_scores(forest_rmse_scores))

##fine tune your model:
##Grid search:
from sklearn.model_selection import GridSearchCV

param_grid = [
	{"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
	{"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5,
				scoring = "neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

#The evaluation scores for each iteration in the random forest 
#algorithm:
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)

##Analyze the best models and their errors:
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

#displaying the importance array with the corresponding names:
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))

##Evaluate you system on the test set:
final_model = grid_search.best_estimator_
x_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline(x_test)
final_prediction = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
