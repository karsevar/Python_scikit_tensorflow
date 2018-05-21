###Hands on machine learning with scikit learn and tensorflow:
##Chapter 5 exercises:

##1.) Support vector machines can be conceptualized as a different form of classification modeling
#algorithm (much like logistic regression). But unlike logistic regression SVMs use the margin
#principles in place of the theta values (from its regression counterparts) to find the optimum
#decision boundary for each dataset. The decision boundary can be separated in two categories 
#soft margin classification and hard margin classification. And the hyperparameter that controls 
#the width of the decision boundary is the C hyperparameter. The C hyperparameter also controls 
#the number of points that are allowed to voilate the decision boundary (which helps the model
#to scale properly to newer data inputs).

#Interestingly the author didn't talk about the maximal margin model (from Introduction to
#statistical learning) which is integral to understanding SVMs and why the C hyperparameter 
#is important. 

##2.) The support vectors are the points on the margins of the decision boundary that determines 
#the margin's location of the decision boundary itself. A larger C parameter brings about 
#wider margins and more decision boundary violations and a smaller C parameter brings about 
#the opposite effects. 

##3.) Much like all of the classification algorithms the X variables being inputted into 
#a support vector machine model needs to be of the same scale. This is import for the algorithm
#to find the best possible decision boundary. 

##4.) The support vector machine method can be considered a black box method in that 
#it can only spit out predictions and not the underlying probabilities that brought 
#about these predictions. For a better model that is more transparent it will be wise 
#to use decision trees or random forests (which has a predict_proba function and 
#the ability to output importance ratings for each of the variables in the model).

##5.) Since this hypothetical situation has more observations than variables (hundreds of 
#variables and millions of instances) the duel problem will work better than the primal 
#problem.

#the author said that the primal problem is the best fit for this problem due to the large
#amount of instances in the hypothetical scenario. Will need to reread this section in chapter
#5.

##6.) Taking from the author's answer, gamma or C should be increased to better fit the training 
#set. 

##8.)
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import make_moons 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score, recall_score
import numpy as np  
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier  

moons = make_moons(n_samples = 1000, shuffle = True, noise = None, random_state = 42)#this will be used to create 
#a linearly separable data set with a linear decision boundary. 

X = moons[0]
y = moons[1] 
X_train = X[200:]
y_train = y[200:]
X_test = X[:200]
y_test = y[:200]

param_grid = [
	{"C": [3, 5, 8, 10, 15, 18, 20, 25], "loss": ["hinge"]},
]

##Linear Support Vector Classifier:
linear_clf_searcher = Pipeline((
	("scaler", StandardScaler()),
	("grid_search", GridSearchCV(LinearSVC(), param_grid, cv = 5, scoring = 
		"accuracy")),
))

linear_clf_searcher.fit(X_train, y_train) 
linear_clf_pred = linear_clf_searcher.predict(X_test)
print(confusion_matrix(linear_clf_pred, y_test))
print(precision_score(y_test, linear_clf_pred))#Precision score is assessed at 0.835
print(recall_score(y_test, linear_clf_pred))# Recall score is assessed at 0.9192 
#the problem with this method is that I don't know what C hyperparameter the grid_search 
#function used for the model. Will need to do this manually. 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train)
clf_single_linear = LinearSVC()
grid_search = GridSearchCV(clf_single_linear, param_grid, cv = 5, scoring = "accuracy")
grid_search.fit(X_train_scaled, y_train)
print(grid_search.best_params_)#The best parameters are loss = hinge and C = 3 (interestingly
#enough).
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(mean_score, params) 

##Support vector classifier:
svc_clf = SVC()
param_grid = [
	{"kernel": ["linear"], "coef0": [1, 2, 3, 4, 5, 10, 15], "C": [0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10, 15, 18, 20, 22, 25, 30]},
]
grid_search_svc = GridSearchCV(svc_clf, param_grid, cv = 5, scoring = "accuracy")
grid_search_svc.fit(X_train_scaled, y_train)
print(grid_search_svc.best_params_)# the best model parameters for this support vector 
#machine method is svc(C = 0.5, coef = 1, kernel = 'linear'). These two models perform 
#achieve the same accuracy rate. 
cvres = grid_search_svc.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(mean_score, params) 
#Testing the model out on the testing set:
X_test_scaled = scaler.transform(X_test)
y_svc_pred = grid_search_svc.predict(X_test)
print(confusion_matrix(y_svc_pred, y_test), precision_score(y_test, y_svc_pred))#On the confusion matrix front the LinearSVC() 
#model performed better with the test dataset.

##Stochastic Gradient Descent classifier:
sgd_clf = SGDClassifier()
param_grid = [
	{"n_iter": [15, 20, 25, 30, 40, 50],"loss": ["hinge"]},
]
grid_search_sgd = GridSearchCV(sgd_clf, param_grid, cv = 5, scoring = "accuracy")
grid_search_sgd.fit(X_train_scaled, y_train)
print(grid_search_sgd.best_params_)
cvres = grid_search_sgd.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(mean_score, params) 
y_sgd_pred = grid_search_sgd.predict(X_test_scaled)
print(confusion_matrix(y_sgd_pred, y_test))
print(precision_score(y_test, y_sgd_pred))#Funny enough from a precision standpoint the 
#stochastic gradient descent model performs the best. 

##10.)
import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit
import python_ML_cat_encoder as en
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer 
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error  

housing = pd.read_csv("/Users/masonkarsevar/housing.csv")
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]
#Setting the training and test sets be the income category:
housing = strat_train_set.drop("median_house_value", axis = 1)
housing_label = strat_train_set["median_house_value"]
for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis = 1, inplace = True)
#Taking out the income_cat column from the dataset.

#the pipelines:
housing_num = housing.drop("ocean_proximity", axis = 1)
housing_cat = housing["ocean_proximity"]
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names 
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values 

num_pipeline = Pipeline([
	("Selector", DataFrameSelector(num_attribs)),
	("imputer", Imputer(strategy = "median")),
	("std_scaler", StandardScaler()),
	])

cat_pipeline = Pipeline([
	("selector", DataFrameSelector(cat_attribs)),
	("cat_encoder", en.CategoricalEncoder(encoding = "onehot-dense")),
	])

full_pipeline = FeatureUnion(transformer_list = [
		("num_pipeline", num_pipeline),
		("cat_pipeline", cat_pipeline),
		])

housing_prepared = full_pipeline.fit_transform(housing)

param_grid_poly = [
	{"kernel": ["poly"], "degree": [1, 2, 3], "C": [15, 20, 25, 30, 35]},
]

param_grid_radial = [{"kernel": ["rbf"], "gamma": [0.01, 0.04, 0.06, 0.1, 0.5, 1, 4, 8, 10, 12, 15], "C": 
	[0.5, 1, 3, 4, 8, 10, 15, 20, 25, 30, 35]},
]
param_grid_linear = [{"kernel": ["linear"], "C": [0.5, 1, 3, 4, 8, 10, 15, 20, 25, 30, 35]},
]
housing_clf = SVR()
grid_search_housing = GridSearchCV(housing_clf, param_grid_poly, scoring = "neg_mean_squared_error")
grid_search_housing.fit(housing_prepared, housing_label)
print(len(housing))
#It seems that this computation is a little too advanced for my computer. But despite that 
#I believe that I have the correct grid search parameters and methodology. I will have 
#to think about a better way to run this computation with the resources that I currently 
#have. 








