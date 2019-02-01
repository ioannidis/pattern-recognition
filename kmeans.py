from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from movielens_data import MovieLensData


dataObj = MovieLensData()
data = dataObj.get_normalized_data()

kmeans = KMeans(n_clusters = 8, precompute_distances = True,)
kmeans.fit(data)  # Fit data into n_clusters

# Getting the cluster labels
labels = kmeans.predict(data)
print(labels)
# Centroid values
centroids = kmeans.cluster_centers_
print(centroids)

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(data[0][0],data[0][1], color='k')
# colmap = ['b','b','b','b','b','b','b','b','b','b']
#
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()
