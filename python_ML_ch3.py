###Hands on machine learning with tensorflow and scikit learning:
##Chapter 3 Classification:
##for this chapter the author will use the MNIST dataset. which 
#is a list of hand written numbers from high school student and 
#US census employees. 

from sklearn.datasets import fetch_mldata 

mnist = fetch_mldata("MNIST original")
print(mnist)

#Datasets loaded by scikit-learn generally have a similar dictionary
#structure including:
    #A DESCR key describing the dataset.
    #A data key containing an array with one row per instance and one
    #column per feature 
    #A target key containing an array with the labels.

x, y = mnist["data"], mnist["target"]
print(x.shape)
print(y.shape)#There are 70,000 images and each image has 784
#features. This is because each image is 28 by 28 pixels, and each
#feature simply represents one pixel's intensity, from 0 and 255.

#Let's display an image from the dataset:
import matplotlib 
import matplotlib.pyplot as plt 

some_digit = x[36000]#Now I understand there are 70,000 images in 
#all and the subset x[36000] is only displaying the 36000th image 
#in the dataset.
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, 
    interpolation = "nearest")
plt.axis("off")
plt.show()

print(y[36000])# The nmist["target"] array are the classification
#labels.

#Paritioning the dataset into a training set and a testing set:
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

#shuffling the training sets:
import numpy as np 

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

##Training a binary classifier:
#To make things more simplier the author will only attempt to 
#classify one instance within the training set (x[36000]) and 
#attempt to see if the classifier will pick one of two levels.
#5 or not 5.

y_train_5 = (y_train == 5) #True for all 5s, false for all other 
#digits.
y_test_5 = (y_test == 5) 

##Starting with a stochastic gradient descent model:
from sklearn.linear_model import SGDClassifier 

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(x_train, y_train_5)# Since the algorithm 
#picks classification values at random (like knn and kmeans) it is
#best to set the seed for reproducibility.
#print(sgd_clf.predict([x[1]]))
#print(y[1])#the classifier is working just fine.

##Performance Measure:
##Measuring Accuracy Using Cross-validation:

##Implementing Cross Validation (from scratch):
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 

skfolds = StratifiedKFold(n_splits=3, random_state = 42)
for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = (y_train_5[train_index])
    x_test_fold = x_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    #print(n_correct / len(y_pred))# this model is very accurate.
    #The cross validation accuracy rate never went below 95 percent.

##k-fold cross validation using the cross_val_score() module:
from sklearn.model_selection import cross_val_score 
print(cross_val_score(sgd_clf, x_train, y_train_5, cv = 3, scoring="accuracy"))

##Dumb classifier algorithm:
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, x, y = None):
        pass 
    def predict(self, x):
        return np.zeros((len(x), 1), dtype = bool)

never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, x_train, y_train_5, cv = 3, scoring = "accuracy"))
#this classifier almost has the same classification accuracy 
#rate as the stochastic gradient descent model above.
#the reason for this is that only 10 percent of the dataset 
#are 5s. 

##Confusion Matrix:
#this will look at the number of times fives were missclassified as
#threes by the classification model. Namely the false positive and
#false negative rate.
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv= 3)
#This command returns the prediction of the model in place 
#of the accuracy ratio.
print(confusion_matrix(y_train_5, y_train_pred))
#precision of the classifier equation:
    #precision = TP/TP + FP

#recall equation:
    #recall = TP / TP + FN

##Precision and Recall:
from sklearn.metrics import precision_score, recall_score 

print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))

##Another metric that combines the precision and recall equations
#is F_1 score (or rather harmonic mean):
from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))

##Precision/Recall tradeoff:
#Setting the decision threshold by hand. Now I understand, this 
#method is very similar to logistical regression (hence the stochastic
#gradient descent name).
#Changing the threshold through the decision_function()

#default threshold:
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

#modified threshold:
threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)
#Increasing the threshold value descreases recall.

##returning decision scores in place of classifications 
#using cross_val_predict()
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv = 3,
    method = "decision_function")
precision, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label = "recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])

plot_precision_recall_vs_threshold(precision, recalls, thresholds)
plt.show()

##precision vs recall:
plt.plot(recalls[:-1], precision[:-1])
plt.ylabel("precision")
plt.xlabel("recall")
plt.show()

##Creating a model with a precision of 90 percent:
y_train_pred_90 = (y_scores > 80000)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

##The ROC Curve:
from sklearn.metrics import roc_curve 

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth=2, label= label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

plot_roc_curve(fpr, tpr)
plt.show()

#calculating the area under the curve. 
from sklearn.metrics import roc_auc_score 

print(roc_auc_score(y_train_5, y_scores))

##Random forest classifier model:
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5,
                    cv = 3, method = "predict_proba")
#Change the probabilities to scores.
y_scores_forest = y_probas_forest[:, 1] #score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, 
                                            y_scores_forest)
plt.plot(fpr, tpr, "b:", label = "SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
#plt.show()

print(roc_auc_score(y_train_5, y_scores_forest))

##Multiclass Classification:
sgd_clf.fit(x_train, y_train)
sgd_clf.predict([some_digit])#This algorithm defaults with the one
#vs one strategy where 9 different binary classifiers are trained
#through the stochastic gradient descent model.
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)
np.argmax(some_digit_scores)
print(sgd_clf.classes_)
print(sgd_clf.classes_[5])

##Overriding the default OneVsOneClassifier or OneVsRestClassifier:
from sklearn.multiclass import OneVsOneClassifier 
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state = 42))
ovo_clf.fit(x_train, y_train)
print(ovo_clf.predict([some_digit]))
print(len(ovo_clf.estimators_))

##RandomForestClassifier:
forest_clf.fit(x_train, y_train)
print(forest_clf.predict([some_digit]))
#the randomforest model already calculates the different 
#probabilities of an instant being apart of the nine different 
#classes.
print(forest_clf.predict_proba([some_digit]))

#checking the accuracy of the stochastic gradient descent model:
print(cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring= "accuracy"))

#Scaling the inputs can increase the accuracy of the model:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
print(cross_val_score(sgd_clf, x_train_scaled, y_train, cv = 3, scoring = "accuracy"))

##Error Analysis:
y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv = 3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

#different visualization options:
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()




















