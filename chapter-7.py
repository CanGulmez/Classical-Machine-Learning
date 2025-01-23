# Ensemble Learning and Random Forests

import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree


# -------------------------------- Voting Classifiers -------------------------


iris = sklearn.datasets.load_breast_cancer()

train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(
   iris["data"], iris["target"], test_size=0.2, random_state=42)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit_transform(train_data)
scaler.fit_transform(test_data)

log_clf = sklearn.linear_model.LogisticRegression()
rnd_clf = sklearn.ensemble.RandomForestClassifier()
svm_clf = sklearn.svm.SVC()

voting_clf = sklearn.ensemble.VotingClassifier(
   estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
   voting="hard"
)

for clf in [log_clf, rnd_clf, svm_clf, voting_clf]:
   clf.fit(train_data, train_labels)
   test_pred = clf.predict(test_data)
   print(clf.__class__.__name__, sklearn.metrics.accuracy_score(test_labels, test_pred))


# ------------------------------- Bagging and Pasting -------------------------


bag_clf = sklearn.ensemble.BaggingClassifier(
   sklearn.tree.DecisionTreeClassifier(), n_estimators=500, max_samples=100,
   bootstrap=True, n_jobs=-1)

bag_clf.fit(train_data, train_labels)

test_pred = bag_clf.predict(test_data)

score = sklearn.metrics.accuracy_score(test_labels, test_pred)
print(score)


# --------------------------- Out-of-Bug Evaluation ---------------------------


bag_clf = sklearn.ensemble.BaggingClassifier(
   sklearn.tree.DecisionTreeClassifier(), n_estimators=500, max_samples=100,
   bootstrap=True, n_jobs=-1, oob_score=True)

bag_clf.fit(train_data, train_labels)

print(bag_clf.oob_score_)

test_pred = bag_clf.predict(test_data)

score = sklearn.metrics.accuracy_score(test_labels, test_pred)
print(score)

print(bag_clf.oob_decision_function_)


# ------------------------------- Random Forests ------------------------------


rnd_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=500,
            max_leaf_nodes=16, n_jobs=-1)

rnd_clf.fit(train_data, train_labels)

test_pred = rnd_clf.predict(test_data)

score = sklearn.metrics.accuracy_score(test_labels, test_pred)
print(score)


# ------------------------------ Boosting -------------------------------------
