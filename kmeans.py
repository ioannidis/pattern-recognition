# Pattern Recognition 2018-2019  #
#================================#
# p16036 - Ioannidis Panagiotis  #
# p16112 - Paravantis Athanasios #
#================================#

from sklearn.cluster import KMeans
from movielens_data import MovieLensData

# Access data
dataObj = MovieLensData()

# Load normalized data
data = dataObj.get_normalized_data()

kmeans = KMeans(n_clusters = 7 , n_init=100, precompute_distances = True,)
kmeans.fit(data)  # Fit data into n_clusters

# Getting the cluster labels
labels = kmeans.predict(data)
print("Labels")
print(labels, "\n")

# Centroid values
centroids = kmeans.cluster_centers_
print("Centroids")
print(centroids)
