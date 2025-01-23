# Dimensionality Reduction

import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import sklearn.datasets
import sklearn.decomposition
import sklearn.manifold
import sklearn.model_selection
import sklearn.preprocessing


# ------------------------------------- PCA -----------------------------------


cancer = sklearn.datasets.load_breast_cancer()

data, labels = cancer["data"], cancer["target"]

scaler = sklearn.preprocessing.StandardScaler()
data = scaler.fit_transform(data)

pca = sklearn.decomposition.PCA(n_components=2)
x2D = pca.fit_transform(data)

print(pca.explained_variance_ratio_)

# Choosing the right of dimensions:
pca = sklearn.decomposition.PCA()
pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

# Better way:
pca = sklearn.decomposition.PCA(n_components=0.95)
reduceddata = pca.fit_transform(data)


# ----------------------------- PCA for Compression ---------------------------


pca = sklearn.decomposition.PCA(n_components=3)
reduced_data = pca.fit_transform(data)
data_recovered = pca.inverse_transform(reduced_data)


# --------------------------------- Randomized PCA ----------------------------


pca = sklearn.decomposition.PCA(n_components=3, svd_solver="randomized")
reduced_data = pca.fit_transform(data)


# ------------------------------- Incremental PCA -----------------------------


n_batches = 100
pca = sklearn.decomposition.IncrementalPCA(n_components=3)
for batch in np.array_split(data, n_batches):
   pca.partial_fit(batch)

reduced_data = pca.transform(data)


# ---------------------------------- Kernel PCA -------------------------------


pca = sklearn.decomposition.KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
reduced_data = pca.fit_transform(data)


# ---------------------------------- LLE --------------------------------------


lle = sklearn.manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=10)
reduced_data = lle.fit_transform(data)
