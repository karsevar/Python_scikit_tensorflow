###Hands on machine learning with scikit learn and tensorflow
##Pretty much this chapter is just like the machine learning 
#stanford coursera course week 1. Will love to see the parallels.

#normal equation check page 108 for more information.
import numpy as np
import matplotlib.pyplot as plt 

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#Computing theta_hat with the normal equation and the preceeding
#values.
#In addition, the function used to generate the data is called 
#Gaussian noise.
x_b = np.c_[np.ones((100, 1)), X] #add x0 = 1 to each instance
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta_best)

#Now we can make predictions using theta_hat:
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance.
y_predict = X_new_b.dot(theta_best)
print(y_predict)


##the following code using linear regression scikit learn:
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

##Gradient Descent
#(interesting note) The MSE cost function for a linear regression
#is a convex function  which means there are no local minimas there
#is only a global minima.

##Batch gradient descent:
#Neat the author is talking about partial derivatives and the cost 
#function technique used by the standford coursera class. very cool.
#batch gradient descent uses the entirety of the train set hence 
#the name. It is better at scaling to datasets with a hight amount 
#of features than the normal equation. 

#Using different learning rates. the author and the coursera lecturer
#was correct learning rate is important for computational speed 
#and global minimum convergence.
eta = 0.1 #name of the learning rate symbol.
n_iterations = 1000
m = 100

#plt.scatter(X, y)
theta = np.random.randn(2,1) #random initialization 
for iteration in range(n_iterations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradients
    x_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new] 
    y_predict = X_new_b.dot(theta)
    #plt.plot(x_new, y_predict, "r-") 

#plt.show()
print(theta)#this algorithm obtained the same values.

##Stochastic gradient descent:
#Unlike batch gradient descent, stochastic gradient descent picks 
#random instance in the training set at every step and computes 
#the gradients based only on that single instance. 
#Irregular minimization of the cost functions. 

#Fixes for this problem are simulated annealing (or rather setting
#a learning schedule that descreases over time).
n_epochs = 50 
t0, t1 = 5, 50# learning schedule hyperparameters.

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1) # random initialization
#plt.scatter(X, y)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch *m + i)
        theta = theta - eta * gradients

        #The following commands are used to plot the 
        #number of iterations on a pyplot 
        x_new = np.array([[0], [2]])
        X_new_b = np.c_[np.ones((2, 1)), X_new] 
        y_predict = X_new_b.dot(theta)
        #plt.plot(x_new, y_predict, "b-")
#plt.show()



##Using the stochastic gradient descent function:
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter = 50, penalty = None, eta0 = 0.1)
#Cool I think the algorithm actually sets the learning schedule 
#automatically.
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)#that's funny the equation
#actually did better than the functions from scikit learn.

##Mini-batch gradient descent:
#computes the gradients on small random sets of instances called 
#mini-batches 
#the author describes the algorithm on page 120. Will need to 
#refer back to that page. 

##Polynomial Regression:
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

##Training quadratic transformation regression with the 
#polynomialfeature class from scikit learn.
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

#X_new_poly = poly_features.fit_transform(array.reshape(range(-3,10)))
    #Will need to look into how to use the reshape method.
#y_pred = lin_reg.predict(X_new_poly)
#plt.scatter(X_new_poly, y)
#plt.plot(X_poly, y_pred, "r-")
#plt.show()

##Learning Curves:
#Finding if a model is overfitted or underfitted using learning 
#curves in place of RMSE and MSE.
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

def plot_learn_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    train_errors, val_errors = [], [] 
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "val")
    plt.ylim((-2, 5))
    plt.show()
lin_reg = LinearRegression()
plot_learn_curves(lin_reg, X, y)

#Creating a polynomial learning curve visualization.
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline((
    ("poly_feature", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
))
plot_learn_curves(polynomial_regression, X, y)
#Interesting for the quadratic regression degree 10 the validation
#set error rate seems to be much better. Will need to look into
#why this is.

##Regularized Linear models:
##Ridge Regression:
#for this chapter the author only talks about the cost function 
#for the ridge regression algorithm. this is really a good refresher
#since I forgot about the sigma value that controls the sinking 
#parameter.

#important note rigde regression is sensitive to different
#x variable scales.
#Interesting I didn't know that you can use ridge regression with 
#a polynomial transformation. Will need to look into the R equivalent.

#Scikit learn using a closed form equation:
from sklearn.linear_model import Ridge 

ridge_reg = Ridge(alpha = 1, solver = "cholesky")
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))

#Using stochastic gradient descent:
sgd_reg = SGDRegressor(penalty = "l2")#look at page 129 to see 
#definition of the penalty = "l2" arguement.
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))

##Lasso Regression:
from sklearn.linear_model import Lasso 

lasso_reg = Lasso(alpha = 0.1)
lasso_reg.fit(X, y)
print(lasso_reg.predict([[1.5]]))

#experiment:
sgd_reg = SGDRegressor(penalty = "l1")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))#That's interesting this prediction
#should be the same as the one above. Will need to look into this.

##Elastic Net:
#this is a combination of both ridge regression and lasso. the main 
#parameter that you have to set with this algorithm is r where 1 
#creates a model that's completely a lasso model and 0 for a 
#completely ridge regression model. 

from sklearn.linear_model import ElasticNet 

elastic_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
elastic_net.fit(X, y)
print(elastic_net.predict([[1.5]]))

##Early Stopping (Using the elbow plot described by machine 
#learning with R):

#early stopping implementation:
from sklearn.base import clone 

sdg_reg = SGDRegressor(n_iter = 1, warm_start=True, penalty = None,
                        learning_rate="constant", eta0=0.0005)
#the warm start argument call just continues training where it left off 
#instead of restarting from scratch. 
#Will work on this problem later. It is located on page 134.

##Logistical regression (binary classification):
#The gold standard posterior probability value is actually set to
#50 percent. 

##Training and cost function:
#look at pages 134 through 135 for the equations and the following
#documentation regarding the mathematical theorems.

##Decision Boundaries:
#For the illustrations the author will use the iris dataset.
from sklearn import datasets 

iris = datasets.load_iris()
print(list(iris.keys()))
X = iris["data"][:, 3:]# pedal width 
y = (iris["target"] == 2).astype(np.int)# 1 if iris-virginica, else 0

from sklearn.linear_model import LogisticRegression 

log_reg = LogisticRegression(random_state = 42)
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth = 2, label = "not Iris Virginica")
plt.show()
#cool way of conceptualizing stigmoid functions. Will need to 
#learn more about the mathematics of these algorithms. 

#This function can also predict full classes through the 50 percent
#decision boundary threshold. 
print(log_reg.predict([[1.7], [1.5]]))#the algorithm gives 
#back a binary answer 1 and 0. 

#logistic regression using two x variables.
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.int)
log_reg2 = LogisticRegression(random_state = 42)
log_reg2.fit(X, y)
x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg2.predict_proba(X_new)

##softmax regression:
#Logistic regression model with more than one class for the response
#variable (namely one vs all and one vs one classifiers).
#The cost function for these kinds of models is called the cross
#entropy equation. Look at the actual definition on page 141.

#It's important to keep in mind that the LogisticRegression function
#uses one vs all if given multiple response classes. To use 
#softmax regression use the multi_class = "multinomial" argument.
X = iris["data"][:, (2,3)] 
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", solver = "lbfgs", C= 10)
softmax_reg.fit(X, y)
print(softmax_reg.predict([[5,2]]))
print(softmax_reg.predict_proba([[5,2]]))

















