###On hands on machine learning with tensorflow and scikit learning.
###chapter 6 Decision Trees:
#Decision trees have much the same functionality as support 
#vector machines (they can be used for classification tasks
#and regression tasks and they can also be coded for multiple
#output models.)

#this chapter will use the CART training algorithm to understand 
#the strengths and short comings of decision trees.

##Training and Visualizing a decision tree:
from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier 

iris = load_iris()
X = iris.data[:, 2:] #This is used to take out the pedal lenght 
#and width.
y = iris.target 

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y) 

#to see the decision tree visualization graphic you can use the 
#export_graphviz module:
from sklearn.tree import export_graphviz 

#export_graphviz(
	#tree_clf, 
	#out_file = image_path("/Users/masonkarsevar/iris_tree.dot"),
	#feature_names=iris.feature_names[2:],
	#class_names=iris.target_names,
	#rounded=True,
	#filled=True
#)
#this command seems to not want to work properly on my system. 
#will go back to this later.

#One important characteristic of this method is that it does not
#require scaling or other data preparation steps.

#the Cart algorithm can only plot two nodes at a time but other 
#decision tree algorithms like the ID3 can plot more than two.

##Estimating Class Probabilities:
print(tree_clf.predict_proba([[5,1.5]]))# much like logistical regression
#models, decision trees use percent confidence in its predictions 
#that a particular instance is a classification class. 
print(tree_clf.predict([[5, 1.5]]))

##The CART Training Algorithm:
#to see the equation for the cost function used to train (or rather
#grow decision tree models) look at page 171. The depth of the 
#tree can be controlled through the max_depth hyperparameter.

##Regularization Hyperparameters:
#Due to the nonparametric nature of decision trees the hyperparameters
#max_depth, min_samples_split, min_samples_leaf, max_samples_leaf,
#etc. are all used to regularize the model. 
#For the complete list of the hyperparameters look at page 174.

##Regression decision trees:
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor
import numpy as np  

np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) **2 
y = y + np.random.randn(m, 1)/10 

tree_reg = DecisionTreeRegressor(max_depth = 2)
tree_reg.fit(X, y)
tree_reg_pred = tree_reg.predict(X)