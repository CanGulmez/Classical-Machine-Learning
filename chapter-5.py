# Support Vector Machines (SVMs)

import numpy as np  
import matplotlib.pyplot as plt 
import sklearn
import sklearn.datasets
import sklearn.preprocessing
import sklearn.svm


# ---------------------------- Linear SVM Classification ----------------------


iris = sklearn.datasets.load_iris()
data = iris["data"][:, (2, 3)]  # petal lenght, petal width
target = (iris["target"] == 2).astype(np.float64) # Irıs-Virginica

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit_transform(data)

lin_svc = sklearn.svm.LinearSVC(loss="hinge", C=1)
lin_svc = sklearn.svm.SVC(kernel="linear", C=1)
lin_svc.fit(data, target)

pred = lin_svc.predict([[5.5, 1.7]])
print(pred)


# ---------------------- Nonlinear SVM Classification -------------------------


poly_feature = sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False)
poly_data = poly_feature.fit_transform(data)

lin_svc = sklearn.svm.LinearSVC(C=10, loss="hinge")
lin_svc.fit(poly_data, target)


# ------------------------------- Polynomial Kernel ---------------------------


poly_svc = sklearn.svm.SVC(kernel="poly", degree=3, coef0=1, C=5)
poly_svc.fit(data, target)


# ---------------------------- Gaussian RBF Kernel ----------------------------


rbf_svc = sklearn.svm.SVC(kernel="rbf", gamma=5, C=0.001)
rbf_svc.fit(data, target)


# ---------------------------------- SVM Regression ---------------------------


lin_svr = sklearn.svm.LinearSVR(epsilon=1.5)
# poly_svr = sklearn.svm.SVR(kernel="poly", degree=3, C=100, epsilon=0.1)
lin_svr.fit(data, target)
