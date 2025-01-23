# Training Models

import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing


# ---------------------------- The Normal Equation ----------------------------


x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

x_b = np.c_[np.ones((100, 1)), x]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new] # add x0 = 1 to each instance
y_predict = x_new_b.dot(theta_best)

print(y_predict)

# plt.plot(x_new, y_predict, "r-")
# plt.plot(x, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()


# ----------------------------- Batch Gradient Descent ------------------------


eta = 0.1   # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) # random initialization

for iteration in range(n_iterations):
   gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)
   theta = theta - eta * gradients

print(theta)


# -------------------------- Stochastic Gradient Descent ----------------------


n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters

def learning_schedule(t):
   return t0 / (t + t1)

theta = np.random.randn(2, 1) # random initialization

for epoch in range(n_epochs):
   for i in range(m):
      random_index = np.random.randint(m)
      xi = x_b[random_index:random_index+1]
      yi = y[random_index:random_index+1]
      gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
      eta = learning_schedule(epoch * m + i)
      theta = theta - eta * gradients

print(theta)


# ----------------------------- Polynomial Regression -------------------------


m = 100
x = 6 * np.random.randn(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

poly_features = sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly_features.fit_transform(x)

print(x[0])
print(x_poly[0])

lin_reg = sklearn.linear_model.LinearRegression()
lin_reg.fit(x_poly, y)

print(lin_reg.intercept_, lin_reg.coef_)

def plot_learning_curves(model, x, y):
   x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, y, 
                                       test_size=0.2)
   train_errors, val_errors = list(), list()
   for m in range(1, len(x_train)):
      model.fit(x_train[:m], y_train[:m])
      y_train_predict = model.predict(x_train[:m])
      y_val_predict = model.predict(x_val)
      train_errors.append(sklearn.metrics.mean_squared_error(y_train[:m], y_train_predict))
      val_errors.append(sklearn.metrics.mean_squared_error(y_val, y_val_predict))
   plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
   plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="val")
   plt.legend()
   plt.show()

lin_reg = sklearn.linear_model.LinearRegression()
plot_learning_curves(lin_reg, x, y)

