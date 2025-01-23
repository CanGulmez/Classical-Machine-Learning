# Classification

import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing


# ---------------------------------- MNIST ------------------------------------


mnist = sklearn.datasets.fetch_openml("mnist_784", version=1)
# print(mnist.keys())

train, target = mnist["data"], mnist["target"]
# print(type(x), type(y))
# print(x.shape, y.shape)

digit = train.iloc[0].to_numpy().reshape(28, 28)
# plt.imshow(digit, cmap=mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

target = target.astype(np.uint8)

train_data, test_data, train_target, test_target = train[:60000], \
   train[60000:], target[:60000], target[60000:]


# ------------------------ Training a Binary Classifier -----------------------


train_target_5 = (train_target == 5)
test_target_5 = (test_target == 5)

sgd_clf = sklearn.linear_model.SGDClassifier(random_state=42)
sgd_clf.fit(train_data, train_target_5)
print(sgd_clf.predict([train.iloc[0]]))

train_target_pred = sklearn.model_selection.cross_val_predict(sgd_clf, 
                        train_data, train_target_5, cv=3)

confusion = sklearn.metrics.confusion_matrix(train_target_5, train_target_pred)
print(confusion)

perfect_pred = train_target_5 # pretend we reached perfection
confusion = sklearn.metrics.confusion_matrix(train_target_5, perfect_pred)
print(confusion)

precision = sklearn.metrics.precision_score(train_target_5, train_target_pred)
print(precision)
recall = sklearn.metrics.recall_score(train_target_5, train_target_pred)
print(recall)
f1 = sklearn.metrics.f1_score(train_target_5, train_target_pred)
print(f1)


# --------------------------- Multiclass Classification -----------------------


sgd_clf.fit(train_data, train_target)

print(sgd_clf.predict([train.iloc[0].to_numpy()]))

forest_clf = sklearn.ensemble.RandomForestClassifier()
forest_clf.fit(train_data, train_target)

print(forest_clf.predict([train.iloc[0].to_numpy()]))
print(forest_clf.predict_proba([train.loc[0].to_numpy()]))

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit_transform(train_data)

scores = sklearn.model_selection.cross_val_score(forest_clf, train_data, 
            train_target, cv=3, scoring="accuracy")
print(scores)

train_target_pred = sklearn.model_selection.cross_val_predict(forest_clf, 
            train_data, train_target, cv=3)

conf_mx = sklearn.metrics.confusion_matrix(train_target, train_target_pred)
print(conf_mx)

# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()


# --------------------------- Multilabel Classification -----------------------


train_target_large = (train_target >= 7)
train_target_odd = (train_target % 2 == 1)
train_target_multilabel = np.c_[train_target_large, train_target_odd]

print(train_target_multilabel)

knn_clf = sklearn.neighbors.KNeighborsClassifier()
knn_clf.fit(train_data, train_target_multilabel)

print(knn_clf.predict([train.iloc[0].to_numpy()]))

train_target_pred = sklearn.model_selection.cross_val_predict(knn_clf, train_data,
            train_target_multilabel, cv=3)

f1 = sklearn.metrics.f1_score(train_target, train_target_pred)
print(f1)
