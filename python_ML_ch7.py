###Hands on machine learning with scikit learn and tensorflow
##chapter 7 Ensemble and Random forests:
#A group of predictors is called an ensemble; thus, this technique is called ensemble learning, and 
#ensemble; thus, this technique is called ensemble learning.

##Voting classifiers:
#Ensemble methods work better when they are independent of one another. Thus meaning 
#partitioning ones data between individual learning models. Will need to get into this.
#Interesting the best way to decrease correlation between the ensemble model is to 
#train the data using different algorithms. 

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

make_moons = make_moons(n_samples = 1000, shuffle = True, noise = 0.2,
    random_state = None)
X = make_moons[0]
X_train = X[:200]
y = make_moons[1]
y_train = y[:200]
X_test = X[200:]
y_test = y[200:]
#This isn't really a good method to parition the data since there are 
#random numbers of 0 and 1 y values within each of the sets. The proportion
#of 0 and 1 values should be uniform between the training set and the testing
#set. 

voting_clf = VotingClassifier(
    estimators = [("Ir", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
    voting = "hard") #this creates a hard voting ensemble model using the 
#logistical regression, random forest, and support vector machine models.
voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#Interesting the random forest classifier and the support vector machine classifier
#both scored a 0.95 accuracy score on the validation set. I really need to look 
#into if this is actually correct. 

##Soft voting is the use of probability to create a decision boundary between 
#the number of y variable classes. Much like logistical regression. This method 
#has higher preformance than hard voting since it gives more weight to higher 
#confidence votes.

svm_clf = SVC(probability = True)#Using the probability hyperparameter in the support 
#vector machine model.
voting_soft_clf = VotingClassifier(
    estimators = [("Ir", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
    voting = "soft")

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#Interestingly the random forest and support vector machines out performed 
#the ensemble soft voting method in this situation.

##Bagging and pasting:
#Both bagging and pasting allow training instances to be sampled several times 
#across multiple predictors, but only bagging allows train instances to be sample 
#severl times for the same predictor.

##Bagging and pasting in scikit learn:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier 

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples = 100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
#the bagging ensemble method has an accuracy rating of 0.957, which is 
#pretty good. 
#this classifier creates 500 different trees for the prediction.

#The Bagging classifier function is set by default to hard voting but you can 
#set the function to soft voting through setting the bootstrap argument to False.

##Out of bag evaluation:
#The out of bag evaluation is more like the default validation set for this method.
#the OOB is not used in training of the classifier and hence there is no need to 
#use cross validation or a validation set to see if the model is working.

#Creating an OOB score through the bagging classifier method:
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators = 500, 
    bootstrap = True, n_jobs = -1, oob_score = True)

bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)#The OOB statistic said that I will likely achieve 
#a 95.5 percent accuracy rate on my validation/test set. Which is interestingly 
#only 0.03 percent off of my accuracy predictions.

#Finding the confidence metrics for each decision made by the model.
print(bag_clf.oob_decision_function_)

##Random patches and Random Samples:
#this function can also sample features within the dataset as well using hard voting 
#and bootstrapping.

#Sampling both training instances and features is called the random patches 
#method and keeping all training instances but sampling the features is called
#the random subset method. 
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators = 500, 
    bootstrap = True, n_jobs = -1, oob_score = True, 
    bootstrap_features = True)

bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)
#Will need to play with this with the baseball dataset from the Lahaman site.

##Random Forests:
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, n_jobs = -1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf. predict(X_test)
print(accuracy_score(y_test, y_pred_rf))# 98 percent accuracy rate this is really good.
#might want to try this with the titanic dataset to obtain a higher score.

##Extra trees:
#You can create an extra tree classifier with the ExtraTreesClassifier.

##Feature importance:
from sklearn.datasets import load_iris
iris = load_iris() 
rnd_clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
#This method can be used in place of the traditional p_value within the linear 
#regression model methodology.

from sklearn.datasets import fetch_mldata 
mnist = fetch_mldata("MNIST original")
X, y = mnist["data"], mnist["target"] 
#letter_clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
#letter_clf.fit(mnist["data"], mnist["target"])
#for name, score in zip(mnist["feature_names"], letter_clf.feature_importances_):
    #print(name, score) 

##Boosting:
#the most popular boosting methods are adaBoost (adaptive boosting) and gradient 
#boosting. 

##Adaboost:
#this method uses the misclassifications of preceeding methods to train 
#later methods. 
#Will need to refer back to pages 192 and 193. this method looks promising.

from sklearn.ensemble import AdaBoostClassifier 

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators = 200,
    algorithm  = "SAMME.R", learning_rate = 0.5)
ada_clf.fit(X_train, y_train)
ada_pred = ada_clf.predict(X_test)
print(accuracy_score(y_test, ada_pred))


##Gradient Boosting:
#this method fits the succeeding model through assessing the preceeding method's 
#residual errors.

#The author will use decision trees to test this method out. This is called 
#Gradient tree boosting.
from sklearn.tree import DecisionTreeClassifier 

tree_reg1 = DecisionTreeClassifier(max_depth=2)
tree_reg1.fit(X_train, y_train)
y2_train = y_train - tree_reg1.predict(X_train)#So this is how you can find the 
#residual values.
tree_reg2 = DecisionTreeClassifier(max_depth = 2)
tree_reg2.fit(X_train, y2_train)
y3_train = y2_train - tree_reg2.predict(X_train)
tree_reg3 = DecisionTreeClassifier(max_depth = 2)
tree_reg3.fit(X_train, y3_train)
y_pred = sum(tree.predict(X_test) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(accuracy_score(y_pred, y_test))
#This isn't really a good implementation of this learning method. Will use a continuous 
#variable instead with the DecisionTreeClassifier 

from sklearn.tree import DecisionTreeRegressor 
import numpy as np 
m = 1000
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X[:, 0]**2 + 0.05 * np.random.randn(m)

tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)#So this is how you can find the 
#residual values.
tree_reg2 = DecisionTreeRegressor(max_depth = 2)
tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth = 2)
tree_reg3.fit(X, y3)
y_pred = sum(tree.predict(X) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(y_pred)

#Gradient boosted random tree methods using sklearn.
from sklearn.ensemble import GradientBoostingRegressor 

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y) 

#A method that allows you to create a Gradient Boosting tree model without the 
#need worry about overfitting the data or underfitting the data.
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 

X_train, X_test, y_train, y_test = train_test_split(X, y)
gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt.staged_predict(X_test)]
bst_n_estimators = np.argmin(errors)
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)

#The following code stops fitting if the mean squared error rate does not improve 
#after five iterations.
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators 
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error 
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break # early breaking 





