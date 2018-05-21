###On hands on machine learning with tensorflow and scikit learning.
###chapter 5 Support Vector machines:
#Support Vector machine are mainly effective with classification 
#problems that are between small and medium sized.

##Linear SVM classification:
#hard margin classification (namely maximal margin classification)
#and soft margin classification (SVM models where the margins are 
#controled through the C hyperparameter).
#A high C parameter gives rise to a smaller margin and a low C 
#parameter brings about the opposite affect.

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler   
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2,3)] #petal length, petal width 
y = (iris["target"] == 2).astype(np.float64)# Iris-virginica 
svm_clf = Pipeline((
		("scaler", StandardScaler()),
		("linear_svc", LinearSVC(C=1, loss = "hinge")),
	))

svm_clf.fit(X, y)
print(svm_clf.predict([[5.5,1.7]]))

##Out of core train alternative:
#SGDClassifier(loss="hinge", alpha = 1/(m*C))

#Addition for better performance set the argument duel
#to false (only if the observations more than the variables).

##Nonlinear SVM classification:
import numpy as np 
from sklearn.datasets import make_moons 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

polynomial_svm_clf = Pipeline((
		("poly_features", PolynomialFeatures(degree = 3)),
		("scaler", StandardScaler()),
		("svm_clf", LinearSVC(C=10, loss="hinge"))
	))

make_moons = datasets.make_moons(n_samples = 100, shuffle = True, noise = 0.15,
	random_state = None)
X = make_moons[0]
y = make_moons[1] 
polynomial_svm_clf.fit(X, y)

def plot_dataset(X, y, axes):
	plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
	plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
	plt.axis(axes)
	plt.grid(True, which = "both")
	plt.xlabel(r"$x_1$", fontsize=20)
	plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

#plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])


def plot_predictions(clf, axes):
	x0s = np.linspace(axes[0], axes[1], 100)
	x1s = np.linspace(axes[2], axes[3], 100)
	x0, x1 = np.meshgrid(x0s, x1s)
	X = np.c_[x0.ravel(), x1.ravel()]
	y_pred = clf.predict(X).reshape(x0.shape)
	y_decision = clf.decision_function(X).reshape(x0.shape)
	plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha = 0.1)
	plt.contourf(x0, x1, y_decision, cmap = plt.cm.brg, alpha = 0.1)

#plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
#plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

#The following commands are from the author's github repo.

##Polynomial kernel (creating polynomial features within a linear
#SVM model without the computational cost):
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline

poly_kernel_clf = Pipeline((
	("scaler", StandardScaler()),
	("svm_clf", SVC(kernel = "poly", degree=10, coef0=100, C =5))
))
#the coef0 argument controls how much the model is influenced by 
#high-degree polynomials versus low-degree polynomials. And of
#course C controls the margin. 
#poly_kernel_clf.fit(X, y)

#plot_predictions(poly_kernel_clf, [-1.5, 2.5, -1, 1.5])
#plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
#plt.show() 

##Adding similarity Features:
#The author describes this method on page 152 will need to come back 
#to this method later.

##Gaussian RBF Kernel (or in the words of introduction to statistical
#learning the radial kernel method):
rbf_kernel_clf = Pipeline((
	("scaler", StandardScaler()),
	("svm_clf", SVC(kernel = "rbf", gamma = 5, C = 1000))
))

rbf_kernel_clf.fit(X, y) 
#plot_predictions(rbf_kernel_clf, [-1.5, 2.5, -1, 1.5])
#plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
#plt.show()

##computational Complexity:
#The LinearSVC class is based on the liblinear library, which implements
#an optimization algorithm for linear SVMs. It does not support the
#kernel trick.

##SVM Regression
#generating random numbers for this representation: 
np.random.seed(42)
m = 50 
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

from sklearn.svm import LinearSVR 

svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)#In place of C 
#the margin hyperparameter is controled with episilon.
svm_reg2 = LinearSVR(epsilon= 0.5, random_state=42)
svm_reg1.fit(X, y)
svm_reg2.fit(X, y)

def find_support_vectors(svm_reg, X, y):
	y_pred = svm_reg.predict(X)
	off_margin = (np.abs(y-y_pred) >= svm_reg.epsilon)
	return np.argwhere(off_margin)

svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

def plot_svm_regression(svm_reg, X, y, axes):
	xls = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
	y_pred = svm_reg.predict(xls)
	plt.plot(xls, y_pred, "k-", linewidth=2, label = r"$\hat{y}$")
	plt.plot(xls, y_pred + svm_reg.epsilon, "k--")
	plt.plot(xls, y_pred - svm_reg.epsilon, "k--")
	plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s = 180, facecolors = "#FFAAAA")
	plt.plot(X, y, "bo")
	plt.axis(axes)

#plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
#plt.show()

##Quadratic transformations with Support vector machine regression.
from sklearn.svm import SVR

svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100,epsilon=0.1)
svm_poly_reg1.fit(X, y)
#plot_svm_regression(svm_poly_reg1, X, y, [-1,1,3,11])
#plt.show()













