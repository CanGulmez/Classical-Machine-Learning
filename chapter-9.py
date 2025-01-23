# Unsupervised Learning Techniques

import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import sklearn.cluster
import sklearn.datasets
import sklearn.decomposition
import sklearn.metrics
import sklearn.neighbors


# ------------------------------------ K-Means --------------------------------


# diabets = sklearn.datasets.load_diabetes()

# data = diabets["data"]

# pca = sklearn.decomposition.PCA(n_components=0.95)
# data = pca.fit_transform(data)

# k = 5 
# kmeans = sklearn.cluster.KMeans(n_clusters=k)
# pred = kmeans.fit_predict(data)

# print(kmeans.labels_)

# print(kmeans.cluster_centers_)

# new_data = np.array([[0, 2, 3, 2, -3, 3, -3, 2.5]])

# print(kmeans.transform(new_data))


# --------------------------------- Mini-batch K-Means ------------------------


minibatch_kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=5)
# minibatch_kmeans.fit(data)


# -------------------- Finding the Optimal Number of Clusters -----------------


# silhouette = sklearn.metrics.silhouette_score(data, kmeans.labels_)
# print(silhouette)


# ------------------- Using Clustering for Image Segmentation -----------------


image = plt.imread("./images/dog.1504.jpg")
# print(image.shape)

data = image.reshape(-1, 3)
kmeans = sklearn.cluster.KMeans(n_clusters=5).fit(data)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)


# -------------------------------------- DBSCAN -------------------------------


data, target = sklearn.datasets.make_moons(n_samples=1000, noise=0.05)
dbscan = sklearn.cluster.DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(data)

print(dbscan.labels_)
print(dbscan.core_sample_indices_)
print(dbscan.components_)

knn = sklearn.neighbors.KNeighborsClassifier(n_jobs=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

new_data = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])

print(knn.predict(new_data))
print(knn.predict_proba(new_data))
