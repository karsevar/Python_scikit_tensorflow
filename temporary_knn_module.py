from sklearn.datasets import fetch_mldata
from sklearn.multiclass import OneVsOneClassifier  
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier  

mnist = fetch_mldata("MNIST original")
x, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = x[:60000], x[100:], y[:60000], y[100:]
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

#knn classifier model using the one vs one classifier method:
knn_num = KNeighborsClassifier(n_jobs=1, weights="distance", n_neighbors = 4)#Since there
#are ten total numbers in the dataset.
knn_num.fit(x_train, y_train)
knn_num_pred = knn_num.predict(x_test)
print(confusion_matrix(knn_num_pred, y_test))