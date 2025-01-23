# End to End Machine Learning Project

import os     
import tarfile
from six.moves import urllib # type: ignore
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.tree


# ------------------------------ Downlaod Data --------------------------------


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
   if not os.path.isdir(housing_path):
      os.makedirs(housing_path)
   tgz_path = os.path.join(housing_path, "housing.tgz")
   urllib.request.urlretrieve(housing_url, tgz_path)
   housing_tgz = tarfile.open(tgz_path)
   housing_tgz.extractall(housing_path)
   housing_tgz.close()

# fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
   csv_path = os.path.join(housing_path, "housing.csv")
   return pd.read_csv(csv_path)


# ------------------- Take a Quick Look at the Data Structure -----------------


housing = load_housing_data()

# print(housing.head())

# print(housing.info())

# print(housing["longitude"])
# print(housing["latitude"])

# print(housing["ocean_proximity"].value_counts())

# print(housing.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()


# ----------------------------- Creating a Test Set ---------------------------

train_set, test_set = sklearn.model_selection.train_test_split(
   housing, test_size=0.2, random_state=42
)

# print(len(train_set))
# print(len(test_set))


# ---------------------- Visualizing Geographical Data ------------------------


# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2)
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10, 7), 
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()


# ------------------------ Looking for Correlations ---------------------------


# droped_housing = housing.drop("ocean_proximity", axis=1)
# corr_matrix = droped_housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


# --------------- Experimenting with Attribute Combinations -------------------


# droped_housing = housing.drop("ocean_proximity", axis=1)
# corr_matrix = droped_housing.corr()

# print(corr_matrix["median_house_value"].sort_values(ascending=False))


# ----------------------------- Data Cleaning ---------------------------------


housing_data = housing.drop("median_house_value", axis=1)
housing_labels = housing["median_house_value"].copy()

# housing_data.dropna(subset=["total_bedrooms"])
# housing_data.drop("total_bedrooms", axis=1)
median = housing_data["total_bedrooms"].median()
housing_data["total_bedrooms"] = housing_data["total_bedrooms"].fillna(median)

housing_data["rooms_per_household"] = housing_data["total_rooms"] / housing_data["households"]
housing_data["bedrooms_per_room"] = housing_data["total_bedrooms"] / housing_data["total_rooms"]
housing_data["population_per_household"] = housing_data["population"] / housing_data["households"]


# ------------------ Handling Text and Categorical Attributes -----------------


# housing_cat = housing_data[["ocean_proximity"]]
# print(housing_cat.value_counts())

encoder = sklearn.preprocessing.OneHotEncoder()

housing_1hot = encoder.fit_transform(housing_data[["ocean_proximity"]])

categories = [category for array in encoder.categories_ for category in array]
encoded = pd.DataFrame(housing_1hot.toarray(), columns=categories)

housing_data = pd.concat([housing_data, encoded], axis=1)

housing_data = housing_data.drop("ocean_proximity", axis=1)


# ---------------------------- Feature Scaling --------------------------------


scaler = sklearn.preprocessing.StandardScaler()

housing_data = scaler.fit_transform(housing_data)
housing_data = pd.DataFrame(housing_data)


# --------------- Training and Evaluating on the Training Set -----------------


train_data = housing_data.iloc[:20000]
test_data = housing_data.iloc[20000:]
train_labels = housing_labels.iloc[:20000]
test_labels = housing_labels.iloc[20000:]

lin_reg = sklearn.linear_model.LinearRegression()
lin_reg.fit(train_data, train_labels)

lin_preds = lin_reg.predict(test_data)

lin_mse = sklearn.metrics.mean_squared_error(test_labels, lin_preds)
lin_rmse = np.sqrt(lin_mse)

print("Linear Regression Error:", lin_rmse)

tree_reg = sklearn.tree.DecisionTreeRegressor()
tree_reg.fit(train_data, train_labels)

tree_preds = tree_reg.predict(test_data)

tree_mse = sklearn.metrics.mean_squared_error(test_labels, tree_preds)
tree_rmse = np.sqrt(tree_mse)

print("Decision Tree Error:", tree_rmse)

forest_reg = sklearn.ensemble.RandomForestRegressor()
forest_reg.fit(train_data, train_labels)

forest_preds = forest_reg.predict(test_data)

forest_mse = sklearn.metrics.mean_squared_error(test_labels, forest_preds)
forest_rmse = np.sqrt(forest_mse)

print("Random Forest Error:", forest_rmse)


# --------------- Better Evaluation Using Cross-Validation --------------------


def display_scores(scores):
   print("Scores:", scores)
   print("Mean:", scores.mean())
   print("Standard deviation:", scores.std())

tree_scores = sklearn.model_selection.cross_val_score(tree_reg, housing_data, 
               housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

display_scores(tree_rmse_scores)

lin_scores = sklearn.model_selection.cross_val_score(lin_reg, housing_data,
               housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)

forest_scores = sklearn.model_selection.cross_val_score(forest_reg, housing_data,
                  housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)


# -------------------- Fine-Tune Your Model: Grid Search ----------------------


param_grid = [
   {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]}, 
   {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

grid_search = sklearn.model_selection.GridSearchCV(forest_reg, param_grid, cv=5, 
                  scoring="neg_mean_squared_error", return_train_score=True)

grid_search.fit(housing_data, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
   print(np.sqrt(-mean_score), params)
