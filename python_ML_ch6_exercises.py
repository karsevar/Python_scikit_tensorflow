###Hands on machine learning with scikit learn and tensorflow:
##Chapter 6 exercises:

##1.) According to the book one can determine the number of branches through the equation 
#log_2(m), where m equals the total number of instances in the dataset. This idea assumes that
#the response variable is binary and that the tree is symetric. According to the author,
#for 1 million instances there will be a total of 20 branches assuming (of course that 
#the tree is symmetric).

##2.) Since the gini coefficient measures node impurity (with zero meaning that the 
#node is completely pure and 1 meaning that the node is completely impure), I can say 
#that the main node will logically be the most impure and the resulting children nodes will
#become increasingly pure. Since decision trees parition observations according to attributes.

##3.) Decreasing max depth would be a very intelligent method to correct for overfitting 
#of a decision tree model.

##4.) Most likely scaling the X variables won't do anything to correct an under performing 
#statistical model. This is because decision trees are one of the few statistical learning 
#method that doesn't require variable scaling. And so the most logical course of action 
#is to increase the max_depth argument within the DecisionTreeClassifier() model call.

##5.) When using the DecisionTreeClassifier() function with the CART training algorithm 
#the computation time can be approximated by the equation O(exp(m)) (where m is the number 
#of instances). With that said though, I really don't know what the O() within the 
#formula stands for. Will need to look this up. 

##6.) The presort parameter will not speed up computation times for datasets with more than 
#a few thousand instances. And so for a model with 100000 different instances setting presort 
#to True will probably do nothing. 

#7.) 
from sklearn.datasets import make_moons 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
import numpy as np 
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples = 10000, noise = 0.4, shuffle = True, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
tree_clf = DecisionTreeClassifier()
param_grid = [
	{"max_depth": [5, 8, 9, 10, 11, 12, 13, 15, 17, 19, 21, 26, 30], "max_leaf_nodes": [None, 2,3,4,5, 6, 7, 8, 9]},
]
grid_search_precision = GridSearchCV(tree_clf, param_grid, scoring = "precision")
grid_search_precision.fit(X_train, y_train) 
print(grid_search_precision.best_params_)
cvres = grid_search_precision.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(mean_score, params) 
#Regardless of what the author said, the best decision tree classifier model has a 
#max depth of 8 with a max leaf node value of none. Will need to check if this is 
#correct. That's not good I'm only getting an precision rating of 84 percent. But then again 
#the author is assessing his model through the use of accuracy.

grid_search_accuracy = GridSearchCV(tree_clf, param_grid, scoring = "accuracy")
grid_search_accuracy.fit(X_train, y_train) 
print(grid_search_accuracy.best_params_)
cvres = grid_search_accuracy.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(mean_score, params)

#Sweet I through this revision I obtained a 85 percent accuracy rate with a max_depth of 5 and 
#a max_leaf_node value of 4. 

##8.) author's solution 
mini_sets = []
split = ShuffleSplit(n_splits = 1000, test_size = len(X_train) - 100, random_state = 42)
for mini_train_index, mini_test_index in split.split(X_train):
	X_mini_train = X_train[mini_train_index]
	y_mini_train = y_train[mini_train_index]
	mini_sets.append((X_mini_train, y_mini_train))

print(mini_sets)#No way this really did create 1000 different subsets of the training 
#set that are 100 in length each. Really cool.

from sklearn.base import clone 

forest = [clone(grid_search_accuracy.best_estimator_) for _ in range(1000)]

accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
	tree.fit(X_mini_train, y_mini_train)

	y_pred = tree.predict(X_test)
	accuracy_scores.append(accuracy_score(y_test, y_pred))

print(np.mean(accuracy_scores))

y_pred = np.empty([1000, len(X_test)], dtype = np.uint8)

for tree_index, tree in enumerate(forest):
	y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode 

y_pred_majority_votes, n_votes = mode(y_pred, axis = 0)
print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))
#will need to use this same method with other classification models. Like the exercise 
#in chapter 5 with a ensemble support vector classifier.











