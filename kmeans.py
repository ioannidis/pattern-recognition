from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# PANDAS ========================================================================

r_cols = ["user id" , "movie id" , "rating", "timestamp"]
ratings = pd.read_csv('./u.data', sep='\t', names=r_cols, encoding='latin-1')

ratings = ratings.sort_values("user id")
ratings = ratings.loc[(ratings["rating"] >= 4)]
# print(ratings)
# print(len(ratings))

# Extract data using pandas
m_cols = ["movie id" , "movie title", "release date" , "video release date" ,
              "IMDb URL" , "unknown" , "Action" , "Adventure" , "Animation" ,
              "Children's" , "Comedy" , "Crime" , "Documentary"
              , "Drama" , "Fantasy" ,
              "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
              "Thriller" , "War" , "Western"]

movies = pd.read_csv('./u.item', sep='|', names=m_cols, encoding='latin-1')
movies = movies.iloc[:, [0] + list(range(6, 24))]
# print(movies)

newArr = pd.merge(ratings,movies, on='movie id')
newArr = newArr.sort_values("user id")
newArrOnlyRatings = newArr.iloc[:, [0] + list(range(4, 22))]
# print(newArrOnlyRatings)

# Data sum values
result = newArrOnlyRatings.groupby(["user id"]).sum()
result = result.iloc[:, list(range(0, 18))]
# print(result)
data = result.values.tolist()
# end Data sum values

# Data  values
# result = newArrOnlyRatings.iloc[:, list(range(1, 19))]
# data = result.values.tolist()
# print(data[0])
# end Data  values

# END PANDAS ========================================================================

kmeans = KMeans(n_clusters=11)
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
