# Pattern Recognition 2018-2019  #
#================================#
# p16036 - Ioannidis Panagiotis  #
# p16112 - Paravantis Athanasios #
#================================#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# PANDAS ========================================================================
theta_list = np.arange(0.1, 2.1, 0.01)

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
# data = result.values.tolist()
# end Data sum values

# Data normalization
norm = pd.DataFrame([])
norm['max_value'] = result.max(axis=1)
norm['min_value'] = result.min(axis=1)

result = result.values.tolist()
norm = norm.values.tolist()

for index, tuple in enumerate(result):
    for i in range(len(tuple)):
        tuple[i] = (tuple[i] - norm[index][1])/(norm[index][0] - norm[index][1])

data = result


# end Data normalization


# END PANDAS ========================================================================

# BSAS =========================================================================================

def bsas(data, theta):
    m                   = 1
    clusters            = []
    cluster_centroids   = []

    clusters.append(data[0])
    cluster_centroids.append(data[0])

    # print(clusters)
    # print(cluster_centroids)

    for vector in data[1:]:
        euclidean_distances = []
        # print("next vector: ", vector)

        for centroid in cluster_centroids:
            euclidean_distances.append(np.linalg.norm(np.subtract(vector, centroid))**2)

        min_euclidean_distance_pos = np.argmin(euclidean_distances)
        # print("euclidean_distances: ", euclidean_distances)
        # print("min_euclidean_distance_pos: ", min_euclidean_distance_pos)
        if euclidean_distances[min_euclidean_distance_pos] > theta:
            m += 1
            clusters.append(vector)
            cluster_centroids.append(vector)
        else:
            clusters[min_euclidean_distance_pos]            = [clusters[min_euclidean_distance_pos], vector]
            cluster_centroids[min_euclidean_distance_pos]   = calculate_centroid(len(clusters[min_euclidean_distance_pos]), cluster_centroids[min_euclidean_distance_pos], vector).tolist()

        # print("=======================================================================================")

    # print("clusters: " , clusters)
    # print("cluster centroids: ", cluster_centroids)
    return [clusters, cluster_centroids]

def calculate_centroid(cluster_size, centroid, vector):
    # print("calculate_centroid parameters: ", cluster_size, centroid, vector)
    # print("calculate_centroid: ", np.true_divide(np.add(np.multiply(cluster_size - 1, centroid), vector), cluster_size))
    return np.add(np.multiply(cluster_size - 1, centroid), vector) / cluster_size


def clusters_number_estimation(theta_list, data):
    cluster_estimations = []

    for theta in theta_list:
        np.random.shuffle(data)
        # print("shuffled data: ", data)
        bsas_result = bsas(data, theta)
        cluster_estimations.append([theta, len(bsas_result[0])])

    with open('your_file.txt', 'w') as f:
        for i in range(len(cluster_estimations)):
            f.write("%.3f %.1f \n" % (theta_list[i], cluster_estimations[i][1]))


    cluster_number_for_each_theta = [number[1] for number in cluster_estimations]
    fig, ax = plt.subplots()
    ax.plot(theta_list, cluster_number_for_each_theta )
    ax.set(xlabel='Theta', ylabel='Number of clusters')
    ax.grid()
    plt.show()



clusters_number_estimation(theta_list, data)

