# Decision Trees

import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import sklearn.datasets
import sklearn.tree


# ------------------ Training and Visualizing a Decision Tree -----------------


iris = sklearn.datasets.load_iris()

x, y = iris.data[:, 2:], iris.target

tree_clf = sklearn.tree.DecisionTreeClassifier()
tree_clf.fit(x, y)

# sklearn.tree.export_graphviz(tree_clf.fit(x, y), out_file="tree.dot", 
#                              rounded=True, filled=True)

pred_prob = tree_clf.predict_proba([[3.5, 0.5]])
print(pred_prob)

pred = tree_clf.predict([[3.5, 0.5]])
print(pred)


# ------------------------------------ Regression -----------------------------


tree_reg = sklearn.tree.DecisionTreeRegressor(max_depth=2)
tree_reg.fit(x, y)
