###Hands on machine learning with scikit learn and tensorflow
##chapter 4 Exercises:

##1.) The best training algorithm for a regression model that has one million features is 
#the gradient descent algorithm since the normal equal is constrained to only datasets that 
#have less features than observations. In addition, dimensional reduction algorithms are need
#for these instances (namely ridge regression, the lasso, and most definitely Principle 
#component analysis).

##2.) The gradient descent algorithm can be affected by differing scales between the X
#variables. The best means to correct this problem is to normalize the X variables 
#(or in other words transform them all into z-scores or zero out the means of each 
#X variable column).

##3.) Since logistical regression is a convex function just like linear regression, one 
#can say that there are no local minima within the equation. There is only a global maxima.
#The only situations where the algorithm might seem to get stuck on a local minima are if 
#one doesn't scale the X variables before initializing the gradient descent algorithm, 
#the learning rate is too small and as such the algorithm may seem to be stuck in a 
#local minima, or the local minima that the algorithm is currently stuck in is the global 
#minima.

##4.) I believe that yes, provided that the learning rates of all the gradient descent experiments
#are all the same and that the number of iterations are especially large, it is possible 
#for stochastic gradient descent, batch gradient descent, and mini batch gradient descent 
#to have close to the same global minima approximations. Almost forgot, the learning rates 
#for mini batch gradient descent and stochastic gradient descent need a learning schedule for 
#them to converge on the global minima and so a possible correction to this answer is if the 
#stochastic and mini batch implementations have the best performing learning rate schedules.

##5.) If the mean squared error value is constantly going up during the implementation of 
#the gradient descent algorithm then that means in place of the cost function converging into 
#the global minima it is diverging. This is primarily caused by a learning rate that is too 
#high for a particular data set. And so the best fix is to replace the learning rate with 
#one that is smaller.

##6.) It is true that mini batch is more stable than stochastic gradient descent, thus giving 
#rise to mean square error values that don't seem to increase quite as much during training.
#Even with that said though, it is important to see to what degree the MSE rate increased. Did 
#it increase a lot or just a little. If it increased by a lot then it is a good idea to stop 
#the algorithm, but if it only increased by a little this only illustrates random variation 
#in the mini batches.

##7.) The fastest gradient descent algorithm is stochastic gradient descent but this is only
#the case for finding the vicinity of the global minima. For the fastest convergence rate,
#the best method is still batch gradient descent (regardless of the algorithm being relatively
#memory inefficient). The best way to make both stochastic and mini batch gradient descent 
#converge faster and well is through computing the best learning rate schedule.

##8.) Most likely the model is over fitted to the training set and so the best way to 
#alleviate this problem is through scaling back on the degree parameter for the quadratic 
#transformation (through the use of grid_search), use k-fold cross validation to find the 
#best degree parameter that will scale well to new data, or graph the MSE values for 
#different degree regimes. 

##9.) Ok I had to look at the author's answer for this question. The regularization term 
#for ridge regression sigma increases bias. This means that a sigma of 1 equals a horizontal
#line going through the data with a high amound of bias. And so the solution to this problem
#is to descrease the sigma hyperparameter.

##10.) 
##a.) Most likely a practicioner will want to use ridge regression in place of standard linear 
#regression (regardless of the sigma hyperparameter being set to 0) is because most likely the 
#person in question will want the future option to shrink extraneous variables within the dataset.
#Since scikit learn doesn't have a p-value option this functionality becomes very useful in the 
#data science python environment.

##b.) A practicioner will want to use lasso above ridge regression because the former zeros
#out all unimportant variables while ridge regression only shrinks the parameter to 1. This 
#functionality gives the lasso method more flexibility can the cost of more variance.

##c.) The elastic Net method is a middle ground between the lasso and ridge regression, thus 
#meaning that you can control the lasso and ridge regression hyperparameters through the 
#hyperparameter r. What makes the elastic net more appealing than the lasso is this middle 
#ground creates a model that has less variance than the lasso and less bias than ridge regression.

##12.) Author's answer:
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression 
import numpy as np 

iris = load_iris()
X = iris["data"][:, (2,3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", C = 10, random_state = 42)
softmax_reg.fit(X, y)
x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 3.5, 200).reshape(-1, 1),
)

X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)
print(y_predict)
print(len(y_predict))

X_with_bias = np.c_[np.ones([len(X), 1]), X]
#this creates the X_0 bias term within the X variable matrix:

np.random.seed(2042)

test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size 
rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

#Interesting each class instance needs its own unique vector for this 
#equation to work.
def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    y_one_hot = np.zeros((m, n_classes))
    y_one_hot[np.arange(m), y] = 1
    return y_one_hot 

print(to_one_hot(y_train[:10]))#this is what he meant will need to look into this
#further.

y_train_one_hot = to_one_hot(y_train)
y_valid_one_hot = to_one_hot(y_valid)
y_test_one_hot = to_one_hot(y_test)

def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis = 1, keepdims = True)
    return exps / exp_sums 
#Now I understand the logit equation is s_k(x) = t(theta^k) * x and so the 
#logit variable is s_k(x) (will need to play with this a little bit).
#For more information look at page 139. 
#And so the softmax function returns the phat_k variable of the equation.

n_inputs = X_train.shape[1] 
n_outputs = len(np.unique(y_train))

eta = 0.01 
n_iterations = 5001 
m = len(X_train)
epsilon = 1e-7

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    y_proba = softmax(logits)
    loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + epsilon), axis = 1))
    error = y_proba - y_train_one_hot 
    if iteration % 500 == 0:
        print(iteration, loss)
    gradients = 1/m * X_train.T.dot(error)
    Theta = Theta - eta * gradients

logits = X_valid.dot(Theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis = 1)
accuracy_score = np.mean(y_predict == y_valid)
print(accuracy_score)
#I still have a long way to go. I really need to brush up on linear algebra, 
# calculous, and computational methods.














