import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.preprocessing import Imputer 
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
import python_ML_cat_encoder as en 
from sklearn.pipeline import FeatureUnion 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder 
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  

##problem 1:
from sklearn.datasets import fetch_mldata
from sklearn.multiclass import OneVsOneClassifier  

mnist = fetch_mldata("MNIST original")
x, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

#knn classifier model using the one vs one classifier method:
knn_num = KNeighborsClassifier(n_neighbors = 10)#Since there
#are ten total numbers in the dataset.
knn_num.fit(x_train, y_train)
knn_num_pred = knn_num.predict(x_test)
print(confusion_matrix(knn_num_pred, y_test))
#My computer is just not powerful enough to run this computation will need to 
#come back to this exercise later on through my studies.


##problem 3:
housing = pd.read_csv("/Users/masonkarsevar/housing.csv")
titanic = pd.read_csv("/Users/masonkarsevar/Desktop/rworks/kaggle_titanic_problem/train.csv")

print(titanic.head())
print(titanic.info())
print(titanic["Cabin"])#Interesting I didn't 
#know that a cabin housed one or two people each. will need to 
#look into this.
print(titanic["Pclass"].value_counts())
print(titanic.describe())

#I will use the stratification sampling method for this problem 
#using the Pclass variable 
split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 42)
for train_index, test_index in split.split(titanic, titanic["Pclass"]):
    strat_train_set = titanic.loc[train_index]
    strat_test_set = titanic.loc[test_index]

print(titanic["Pclass"].value_counts()/len(titanic))
print(strat_train_set["Pclass"].value_counts()/len(strat_train_set))
print(strat_test_set["Pclass"].value_counts()/len(strat_test_set))
#Sweet the proportions are all the same for the test and training sets.

#will need to drop the names and the embarked variables.
for set_ in (strat_train_set, strat_test_set):
    set_.drop(["Embarked","Ticket","Name"], axis = 1, inplace = True)

titanic_variable = strat_train_set.drop(["Survived","PassengerId"], axis = 1)
titanic_id = strat_train_set["PassengerId"].copy()#Need to separate 
#the passengerids from the regression or KNN models as well. 
titanic_label = strat_train_set["Survived"].copy() 

##encoding the categorical variables within the dataset 
titanic_variable_num = titanic_variable.drop(["Sex","Pclass","Cabin"], axis = 1)
titanic_variable_cat = titanic_variable.drop(["Age","SibSp","Parch","Fare","Cabin"], axis = 1)
#do to the NaN values in the cabin variable, we can't use the variable within
#this function call.
num_attribs = list(titanic_variable_num)
cat_attribs = list(titanic_variable_cat)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attribs)),
    ("imputer", Imputer(strategy = "median")),
    ("std_scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attribs)),
    ("encoder", en.CategoricalEncoder(encoding = "onehot-dense")),
])

full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

titanic_prepared = full_pipeline.fit_transform(titanic_variable)
print(titanic_prepared.shape)

titanic_svm = LinearSVC(C=1, loss = "hinge")
titanic_svm.fit(titanic_prepared, titanic_label)
titanic_pred = titanic_svm.predict(titanic_prepared)
print(confusion_matrix(titanic_label, titanic_pred))

titanic_test_prepared = full_pipeline.fit_transform(strat_test_set)
titanic_pred = titanic_svm.predict(titanic_test_prepared)
print(confusion_matrix(strat_test_set["Survived"], titanic_pred))

#Automation idea for this problem:
model_pipeline = Pipeline([
    ("full_pipeline", full_pipeline),
    ("Grid_search", GridSearchCV(LinearSVC(), [{"C": [1, 5, 15, 20, 25, 30, 35, 40], "loss":["hinge"],}],
        cv = 10, scoring = "accuracy")),
])
model_pipeline.fit(titanic_variable, titanic_label)
titanic_pred = model_pipeline.predict(titanic_variable)
print(confusion_matrix(titanic_label, titanic_pred))
#Not a bad idea but I think gradient descent will work a little better.

#Using the LinearSVC method:
param_grid = [
    {"C": [1,5,10, 15, 20, 25], "loss": ["hinge"]},
]
svc = LinearSVC()
grid_search = GridSearchCV(svc, param_grid, cv = 5, scoring = "accuracy")
grid_search.fit(titanic_prepared, titanic_label)
print(grid_search.best_params_)
crves = grid_search.cv_results_
for mean_score, params in zip(crves["mean_test_score"], crves["params"]):
    print(mean_score, params)
#the best model for the LinearSVC method is LinearSVC(C=25, loss = "hinge")
# let's see if other methods can do any better. 

#Using support vector machines with a radial transformation:
from sklearn.svm import SVC 
svc_radial = SVC()
param_grid = [
    {"kernel": ["rbf"], "C": [1, 10, 20, 30], "gamma": [0.1, 0.3, 0.4, 2, 3]},
] 
grid_search = GridSearchCV(svc_radial, param_grid, scoring = "average_precision")
grid_search.fit(titanic_prepared, titanic_label)
print(grid_search.best_params_)
crves = grid_search.cv_results_
for mean_score, params in zip(crves["mean_test_score"], crves["params"]):
    print(mean_score, params)
#validation set:
titanic_test_vars = strat_test_set.drop(["Survived","Cabin"], axis = 1)
titanic_test_labels = strat_test_set["Survived"].copy()
titanic_test_prepared = full_pipeline.transform(titanic_test_vars)
titanic_pred_val = grid_search.predict(titanic_test_prepared)
print(confusion_matrix(titanic_pred_val, titanic_test_labels))
#this model has a 79 percent true positive and negative rate.

#Logistic regression model:
log_reg = LogisticRegression() 
log_reg.fit(titanic_prepared, titanic_label)
titanic_pred_log = log_reg.predict(titanic_test_prepared)
print(confusion_matrix(titanic_pred_log, titanic_test_labels))
#It only trails the support vector machine method by only 8 miss classifications.

#Soft max logistic regression:
soft_max = LogisticRegression()
param_grid = [
    {"solver": ["lbfgs"], "C": [20, 25, 30, 35, 40, 45, 50]},
]
grid_search = GridSearchCV(soft_max, param_grid, scoring = "average_precision")
grid_search.fit(titanic_prepared, titanic_label)
crves = grid_search.cv_results_
for mean_score, params in zip(crves["mean_test_score"], crves["params"]):
    print(mean_score, params)
#validation set:
titanic_soft_pred = grid_search.predict(titanic_test_prepared)
print(confusion_matrix(titanic_soft_pred, titanic_test_labels))
#this method preformed similar to the normal logistical regression model.

#Random forest:
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state = 42)
forest_clf.fit(titanic_prepared, titanic_label)
forest_pred = forest_clf.predict(titanic_test_prepared)
print(confusion_matrix(forest_pred, titanic_test_labels))
#The random forest model has success rate of 0.7988 which is a little better than 
#the support vector machine classifiers.









